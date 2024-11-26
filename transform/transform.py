import itertools
import logging

import numpy as np
import pandas as pd
import xarray as xr
from pandas import timedelta_range, date_range
from xarray import DataArray

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, UserType, BillType
from operation.definitions import Status
from operation.scale_profile import ProportionateScaler, ScaleTimeOfUseProfile
from utility import configuration

logger = logging.getLogger(__name__)


class DataTransformer(PipelineStage):
    stage = Stage.TRANSFORM
    _name = "data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class UserDataTransformer(DataTransformer):
    _name = "user_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dataset.loc[..., ColumnName.USER_TYPE] = xr.apply_ufunc(lambda x: UserType(x),
                                                                dataset.sel({"dim_1": ColumnName.USER_TYPE}),
                                                                vectorize=True)
        dataset.loc[..., [ColumnName.USER_ADDRESS, ColumnName.DESCRIPTION]] = xr.apply_ufunc(lambda x: x.strip(),
                                                                                             dataset.sel({"dim_1": [
                                                                                                 ColumnName.USER_ADDRESS,
                                                                                                 ColumnName.DESCRIPTION]}),
                                                                                             vectorize=True)
        return dataset


class BillDataTransformer(DataTransformer):
    _name = "bill_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        def get_bill_type(df):
            return BillType.TIME_OF_USE if any(df.loc[:, ColumnName.MONO_TARIFF].isnull()) else BillType.MONO

        da = DataArray(list(itertools.chain.from_iterable(
            [get_bill_type(df), ] * df.shape[0] for _, df in dataset.groupby(dataset.loc[:, ColumnName.USER]))))
        da = da.expand_dims("dim_1").assign_coords({"dim_1": [ColumnName.BILL_TYPE, ]})
        dataset = xr.concat([dataset, da], dim="dim_1").rename({"dim_1": ColumnName.USER_DATA.value})
        return dataset


class ReshaperByYear(DataTransformer):
    _name = "year_reshaper"

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        user_name = kwargs.pop('user_name')
        ref_year = configuration.config.getint("time", "year")
        dims = (ColumnName.DATE.value, ColumnName.TIME.value, ColumnName.USER.value)
        coords = {ColumnName.DATE.value: date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"),
                  ColumnName.TIME.value: timedelta_range(start="0 Days",
                                                         freq=configuration.config.get("time", "resolution"),
                                                         periods=dataset.shape[1]), ColumnName.USER.value: [1, ]}
        dataset = OmnesDataArray(dataset.values, dims=dims, coords=coords)
        dataset[ColumnName.USER.value] = user_name
        return dataset


class TypicalLoadProfileTransformer(DataTransformer):
    _name = "typical_load_profile_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset.loc[..., ColumnName.USER_TYPE] = xr.apply_ufunc(lambda x: UserType(x),
                                                                dataset.sel({"dim_1": ColumnName.USER_TYPE}),
                                                                vectorize=True)
        values = dataset.where(~dataset.dim_1.isin((ColumnName.USER_TYPE, ColumnName.MONTH)), drop=True)
        day_types = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[1].strip("j")), values.dim_1,
                                   vectorize=True)
        hour = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), values.dim_1,
                              vectorize=True)
        values.reset_index(dims_or_levels=("dim_0", "dim_1"))

        dims = (ColumnName.DAY_TYPE.value, ColumnName.USER_TYPE.value, ColumnName.HOUR.value, ColumnName.MONTH.value)
        coords = {ColumnName.DAY_TYPE.value: np.unique(day_types), ColumnName.HOUR.value: np.unique(hour),
                  ColumnName.USER_TYPE.value: np.unique(dataset.loc[:, ColumnName.USER_TYPE]),
                  ColumnName.MONTH.value: np.unique(dataset.loc[:, ColumnName.MONTH]) + 1}
        new_array = OmnesDataArray(dims=dims, coords=coords)
        for user_type, df in values.groupby(dataset.loc[:, ColumnName.USER_TYPE]):
            hours = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), df.dim_1,
                                   vectorize=True)
            for hour, df in df.groupby(hours):
                new_array.loc[:, user_type, hour, :] = df.values.T
        return new_array


class PvPlantDataTransformer(DataTransformer):
    _name = "pv_plant_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return dataset.rename({"dim_1": ColumnName.USER_DATA.value})


class ProductionDataTransformer(DataTransformer):
    _name = "pv_production_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return dataset


class TariffTransformer(DataTransformer):
    _name = "tariff_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset = dataset.rename({"dim_0": ColumnName.DAY_TYPE.value, "dim_1": ColumnName.HOUR.value})
        dataset = dataset - 1
        dataset[ColumnName.DAY_TYPE.value] = dataset[ColumnName.DAY_TYPE.value].astype(int)
        dataset[ColumnName.HOUR.value] = dataset[ColumnName.HOUR.value].astype(int)
        return dataset


class BillLoadProfileTransformer(DataTransformer):
    @classmethod
    def get_time_of_use_labels(cls, bill_type):
        if bill_type == BillType.MONO:
            return ColumnName.MONO_TARIFF
        elif bill_type == BillType.TIME_OF_USE:
            return configuration.config.getarray("tariff", "time_of_use_labels", str, fallback=None)
        else:
            raise ValueError(f"Invalid bill_type '{bill_type}'")

    _name = "bill_load_profile_transformer"
    _profile_scaler = {BillType.MONO: ProportionateScaler(), BillType.TIME_OF_USE: ScaleTimeOfUseProfile()}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        data_store = DataStore()
        typical_load_profiles = data_store["typical_load_profiles_gse"]
        user_profiles = OmnesDataArray(
            dims=(ColumnName.USER.value, ColumnName.MONTH.value, ColumnName.DAY_TYPE.value, ColumnName.HOUR.value),
            coords={ColumnName.USER.value: np.unique(dataset.sel({ColumnName.USER_DATA.value: ColumnName.USER})),
                    ColumnName.MONTH.value: np.unique(dataset.sel({ColumnName.USER_DATA.value: ColumnName.MONTH})),
                    ColumnName.DAY_TYPE.value: configuration.config.getarray("time", "day_types", int),
                    ColumnName.HOUR.value: range(24)})

        grouper = xr.DataArray(pd.MultiIndex.from_arrays(
            [dataset.sel(user_data=ColumnName.MONTH).values, dataset.sel(user_data=ColumnName.USER).values],
            names=['month', 'user'], ), dims=['dim_0'], coords={"dim_0": dataset.dim_0.values}, )
        for (month, user), ds in dataset.groupby(grouper):
            selection = {ColumnName.USER_TYPE.value: UserType.PDMF, ColumnName.MONTH.value: month}
            reference_profile = typical_load_profiles.sel(selection).squeeze()
            aggregated_consumption_of_reference_profile = DataStore()["typical_aggregated_consumption"].sel(
                selection).squeeze()
            user_profiles.loc[user, month, :, :] = self.scale_profile(ds.sel(user_data=ColumnName.BILL_TYPE).values[0],
                                                                      ds, reference_profile,
                                                                      aggregated_consumption_of_reference_profile)
        return user_profiles

    @classmethod
    def scale_profile(cls, bill_type, bill, *args, **kwargs):
        # TODO: turn this part into subclasses instead of dictionaries
        scaler = cls._profile_scaler[bill_type]
        scaled_profile = scaler(bill.sel(user_data=cls.get_time_of_use_labels(bill_type)), *args, **kwargs)
        if scaler.status not in (Status.OPTIMAL, Status.OK):
            logger.warning(f"Load profile scaler returned with invalid status: {scaler.status}")
        return scaled_profile


class YearlyProfileCreator(DataTransformer):
    _name = "yearly_profile_creator"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        ref_year = configuration.config.getint("time", "year")
        output_data = OmnesDataArray(0., dims=(ColumnName.USER.value, ColumnName.TIME.value), coords={
            ColumnName.USER.value: np.unique(dataset.sel({ColumnName.USER_DATA.value: ColumnName.USER})),
            ColumnName.TIME.value: date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31",
                                              freq=configuration.config.get("time", "resolution"))})
        for user, df in user_data.groupby(ColumnName.USER):
            # Evaluate profiles in typical days
            months = np.repeat(df.loc[:, ColumnName.MONTH], ni)
            day_types = np.repeat(df.loc[:, ColumnName.DAY_TYPE], ni)
            profiles = df.loc[:, 0:].values.flatten()
            profiles = ProfileExtractor.create_typical_profile_from_yearly_profile(profiles, months, day_types).reshape(
                (nm, nj * ni))
            # Evaluate typical profiles in each month
            nds = df.groupby([ColumnName.MONTH, ColumnName.DAY_TYPE]).count().iloc[:, 0].values.reshape(nm, nj)
            tou_energy = []
            for y, nd in zip(profiles, nds):
                tou_energy.append(ProfileExtractor.get_monthly_consumption(y, nd))
            # Create dataframe
            tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)), axis=1)
            tou_energy = DataFrame(tou_energy,
                                   columns=configuration.config.getarray("tariff", "time_of_use_labels", str))
            tou_energy.insert(0, ColumnName.USER, user)
            tou_energy.insert(1, ColumnName.YEAR, configuration.config.getint("time", "year"))
            tou_energy.insert(2, ColumnName.MONTH, ms)
            # Concatenate
            output_df = concat((output_df, tou_energy), axis="rows")
        return output_df


class ProfileDataAggregator(DataTransformer):
    _name = "profile_data_aggregator"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return dataset
