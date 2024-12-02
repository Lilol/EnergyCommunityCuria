import itertools
import logging

import numpy as np
import pandas as pd
import xarray as xr
from pandas import timedelta_range, date_range, to_datetime
from xarray import DataArray

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind, UserType, BillType
from operation.definitions import Status
from operation.scale_profile import ProportionateScaler, ScaleTimeOfUseProfile
from utility import configuration
from utility.day_of_the_week import get_weekday_code
from utility.definitions import grouper

logger = logging.getLogger(__name__)


class Transform(PipelineStage):
    stage = Stage.TRANSFORM
    _name = "data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class TransformReferenceProfile(PipelineStage):
    _name = "ref_profile_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dataset = dataset.rename({"dim_1": DataKind.USER_DATA.value}).assign_coords(
            {"date": to_datetime(dataset.date.date, format="ISO8601").floor("h")})
        da = OmnesDataArray(xr.apply_ufunc(get_weekday_code, dataset.date.dt.date, vectorize=True),
                            dims=DataKind.DATE.value, coords={DataKind.DATE.value: dataset.date}).expand_dims(
            DataKind.USER_DATA.value).assign_coords({DataKind.USER_DATA.value: [DataKind.DAY_TYPE, ]})
        dataset = xr.concat([dataset, da], dim=DataKind.USER_DATA.value)

        typical_profiles = None
        for user_type in UserType:
            if user_type == UserType.PV:
                continue
            da = xr.concat([xr.concat([dk.sel(user_data=user_type).groupby(dk.date.dt.hour).mean(skipna=True,
                                                                                                 keep_attrs=False).assign_coords(
                hour=[f'y_j{int(dt)}_i{int(h)}' for h in range(24)], month=int(m)).squeeze(drop=True) for dt, dk in
                                       dk.groupby(dk.sel({DataKind.USER_DATA.value: DataKind.DAY_TYPE}))],
                                      dim=DataKind.HOUR.value).squeeze(drop=True) for m, dk in
                dataset.groupby(dataset.sel({DataKind.USER_DATA.value: DataKind.MONTH}))], dim="type").assign_coords(
                type=[user_type.value] * 12).drop({DataKind.USER_DATA.value: user_type})
            # TODO: add month info to frame as a column
            typical_profiles = xr.concat([typical_profiles, da], dim="type") if typical_profiles is not None else da
        return typical_profiles.set_index(type="type")


class TransformUserData(Transform):
    _name = "user_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dataset = dataset.rename({"dim_1": DataKind.USER_DATA.value, "dim_0": DataKind.USER.value}).assign_coords(
            {DataKind.USER.value: dataset.sel({"dim_1": DataKind.USER}).squeeze().values}).drop(labels=DataKind.USER,
                                                                                                dim=DataKind.USER_DATA.value)
        dataset.loc[..., DataKind.USER_TYPE] = xr.apply_ufunc(lambda x: UserType(x), dataset.sel(
            {DataKind.USER_DATA.value: DataKind.USER_TYPE}), vectorize=True)
        dataset.loc[..., [DataKind.USER_ADDRESS, DataKind.DESCRIPTION]] = xr.apply_ufunc(lambda x: x.strip(),
                                                                                         dataset.sel({
                                                                                             DataKind.USER_DATA.value: [
                                                                                                 DataKind.USER_ADDRESS,
                                                                                                 DataKind.DESCRIPTION]}),
                                                                                         vectorize=True)
        return dataset


class TransformBills(Transform):
    _name = "bill_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        def get_bill_type(df):
            return BillType.TIME_OF_USE if any(df.loc[..., DataKind.MONO_TARIFF].isnull()) else BillType.MONO

        da = DataArray(list(itertools.chain.from_iterable(
            [get_bill_type(df.squeeze(drop=True)), ] * df.shape[1] for _, df in
            dataset.groupby(dataset.squeeze().loc[..., DataKind.USER]))))
        da = da.expand_dims("dim_1").assign_coords({"dim_1": [DataKind.BILL_TYPE, ]})
        dataset = (xr.concat([dataset, da], dim="dim_1").rename(
            {"dim_1": DataKind.USER_DATA.value, "dim_0": DataKind.USER.value}))

        dataset = dataset.assign_coords(
            {DataKind.USER.value: dataset.sel({DataKind.USER_DATA.value: DataKind.USER}).squeeze().values}).drop(
            labels=DataKind.USER, dim=DataKind.USER_DATA.value)

        return dataset


class ReshaperByYear(Transform):
    _name = "year_reshaper"

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        user_name = kwargs.pop('user_name')
        ref_year = configuration.config.getint("time", "year")
        dims = (DataKind.DATE.value, DataKind.TIME.value, DataKind.USER.value)
        coords = {DataKind.DATE.value: date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"),
                  DataKind.TIME.value: timedelta_range(start="0 Days",
                                                       freq=configuration.config.get("time", "resolution"),
                                                       periods=dataset.shape[1]), DataKind.USER.value: [1, ]}
        dataset = OmnesDataArray(dataset.values, dims=dims, coords=coords)
        dataset[DataKind.USER.value] = user_name
        return dataset


class TransformTypicalLoadProfile(Transform):
    _name = "typical_load_profile_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset.loc[..., DataKind.USER_TYPE] = xr.apply_ufunc(lambda x: UserType(x),
                                                              dataset.sel({"dim_1": DataKind.USER_TYPE}),
                                                              vectorize=True)
        values = dataset.where(~dataset.dim_1.isin((DataKind.USER_TYPE, DataKind.MONTH)), drop=True)
        day_types = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[1].strip("j")), values.dim_1,
                                   vectorize=True)
        hour = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), values.dim_1,
                              vectorize=True)
        values.reset_index(dims_or_levels=("dim_0", "dim_1"))

        dims = (DataKind.DAY_TYPE.value, DataKind.USER_TYPE.value, DataKind.HOUR.value, DataKind.MONTH.value)
        coords = {DataKind.DAY_TYPE.value: np.unique(day_types), DataKind.HOUR.value: np.unique(hour),
                  DataKind.USER_TYPE.value: np.unique(dataset.loc[:, DataKind.USER_TYPE]),
                  DataKind.MONTH.value: np.unique(dataset.loc[:, DataKind.MONTH]) + 1}
        new_array = OmnesDataArray(dims=dims, coords=coords)
        for user_type, df in values.groupby(dataset.loc[:, DataKind.USER_TYPE]):
            hours = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), df.dim_1,
                                   vectorize=True)
            for hour, df in df.groupby(hours):
                new_array.loc[:, user_type, hour, :] = df.values.T
        return new_array


class TransformPvPlantData(Transform):
    _name = "pv_plant_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dataset = dataset.rename({"dim_1": DataKind.USER_DATA.value, "dim_0": DataKind.USER.value})

        dataset = dataset.assign_coords(
            {DataKind.USER.value: dataset.sel({DataKind.USER_DATA.value: DataKind.USER}).squeeze().values}).drop(
            labels=DataKind.USER, dim=DataKind.USER_DATA.value)

        return dataset


class TransformProduction(Transform):
    _name = "pv_production_data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return dataset


class TransformTariffData(Transform):
    _name = "tariff_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset = dataset.rename({"dim_0": DataKind.DAY_TYPE.value, "dim_1": DataKind.HOUR.value})
        dataset = dataset - 1
        dataset[DataKind.DAY_TYPE.value] = dataset[DataKind.DAY_TYPE.value].astype(int)
        dataset[DataKind.HOUR.value] = dataset[DataKind.HOUR.value].astype(int)
        return dataset


class TransformBillsToLoadProfiles(Transform):
    @classmethod
    def get_time_of_use_labels(cls, bill_type):
        if bill_type == BillType.MONO:
            return DataKind.MONO_TARIFF
        elif bill_type == BillType.TIME_OF_USE:
            return configuration.config.getarray("tariff", "time_of_use_labels", str, fallback=None)
        else:
            raise ValueError(f"Invalid bill_type '{bill_type}'")

    _name = "bill_load_profile_transformer"
    _profile_scaler = {BillType.MONO: ProportionateScaler(), BillType.TIME_OF_USE: ScaleTimeOfUseProfile()}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        user_type = kwargs.pop("user_type", None)
        typical_load_profiles = DataStore()["typical_load_profiles_gse"]
        users = DataStore()["users"]
        user_profiles = OmnesDataArray(dims=(
            DataKind.USER.value, DataKind.MONTH.value, DataKind.DAY_TYPE.value, DataKind.HOUR.value,
            DataKind.MUNICIPALITY.value), coords={DataKind.USER.value: np.unique(dataset[DataKind.USER.value]),
                                                  DataKind.MONTH.value: np.unique(
                                                      dataset.sel({DataKind.USER_DATA.value: DataKind.MONTH})),
                                                  DataKind.DAY_TYPE.value: configuration.config.getarray("time",
                                                                                                         "day_types",
                                                                                                         int),
                                                  DataKind.HOUR.value: range(24),
                                                  DataKind.MUNICIPALITY.value: dataset[DataKind.MUNICIPALITY.value]})

        for m, ds in dataset.groupby(DataKind.MUNICIPALITY.value):
            groups = grouper(ds, DataKind.USER.value, user_data=DataKind.MONTH)
            for (user, month), ds in ds.groupby(groups):
                if user_type is None:
                    ut = users.sel({DataKind.MUNICIPALITY.value: m, DataKind.USER_DATA.value: DataKind.USER_TYPE,
                                    DataKind.USER.value: user}).values
                else:
                    ut = user_type
                selection = {DataKind.USER_TYPE.value: ut, DataKind.MONTH.value: month}
                reference_profile = typical_load_profiles.sel(selection).squeeze()
                aggregated_consumption_of_reference_profile = DataStore()["typical_aggregated_consumption"].sel(
                    selection).squeeze()
                user_profiles.loc[user, month, ..., m] = self.scale_profile(
                    ds.sel(user_data=DataKind.BILL_TYPE).values[0, 0], ds, reference_profile,
                    aggregated_consumption_of_reference_profile)
        return user_profiles

    @classmethod
    def scale_profile(cls, bill_type, bill, *args, **kwargs):
        # TODO: turn this part into subclasses instead of dictionaries
        scaler = cls._profile_scaler[bill_type]
        scaled_profile = scaler(bill.sel(user_data=cls.get_time_of_use_labels(bill_type)), *args, **kwargs)
        if scaler.status not in (Status.OPTIMAL, Status.OK):
            logger.warning(f"Load profile scaler returned with error status: '{scaler.status}'")
        return scaled_profile


class CreateYearlyProfile(Transform):
    _name = "yearly_profile_creator"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        ref_year = configuration.config.getint("time", "year")
        output_data = OmnesDataArray(0., dims=(DataKind.USER.value, DataKind.TIME.value, DataKind.MUNICIPALITY.value),
                                     coords={DataKind.USER.value: np.unique(dataset[DataKind.USER.value]),
                                             DataKind.TIME.value: date_range(start=f"{ref_year}-01-01",
                                                                             end=f"{ref_year + 1}-01-01",
                                                                             freq=configuration.config.get("time",
                                                                                                           "resolution"),
                                                                             inclusive="left"),
                                             DataKind.MUNICIPALITY.value: dataset[DataKind.MUNICIPALITY.value]})
        for day in date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"):
            # Retrieve profiles in typical days
            output_data.loc[:, f"{day:%Y-%m-%d}", :] = dataset.sel(
                {DataKind.DAY_TYPE.value: get_weekday_code(day), DataKind.MONTH.value: day.month}).values
        return output_data


class AggregateProfileDataForTimePeriod(Transform):
    _name = "profile_data_aggregator"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        tou_labels = configuration.config.get("tariff", "time_of_use_labels")
        output_data = OmnesDataArray(0.,
                                     dims=(DataKind.USER.value, DataKind.TOU_ENERGY.value, DataKind.MUNICIPALITY.value),
                                     coords={DataKind.USER.value: dataset[DataKind.USER.value],
                                             DataKind.TOU_ENERGY.value: [*tou_labels, DataKind.ANNUAL_ENERGY],
                                             DataKind.MUNICIPALITY.value: np.unique(
                                                 dataset[DataKind.MUNICIPALITY.value])})
        tou_time_slots = DataStore()["time_of_use_time_slots"]
        tou_time_slot_values = configuration.config.getarray("tariff", "tariff_time_slots", int)
        for day, ds in dataset.groupby(dataset.time.dt.dayofyear):
            day = pd.to_datetime(ds.time.values[0])
            ds = ds.assign_coords({DataKind.TIME.value: ds.time.dt.hour}).rename(
                {DataKind.TIME.value: DataKind.HOUR.value})
            output_data.loc[:, tou_labels, :] += xr.concat([ds.sel({DataKind.HOUR.value: tou_time_slots.sel(
                {DataKind.DAY_TYPE.value: get_weekday_code(day)}) == tou_time_slot}).sum(
                dim=DataKind.HOUR.value).assign_coords(
                {DataKind.ANNUAL_ENERGY.value: f"energy{tou_time_slot + 1}"}).squeeze(drop=True) for tou_time_slot in
                                                            tou_time_slot_values], dim=DataKind.ANNUAL_ENERGY.value)
        output_data.loc[:, DataKind.ANNUAL_ENERGY] = output_data.loc[:, tou_labels].sum(dim="energy")
        return output_data


class TransformTimeOfUseTimeSlots(Transform):
    _name = "tou_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return xr.concat([OmnesDataArray(unique_numbers[1], dims=DataKind.COUNT.value,
                                         coords={DataKind.COUNT.value: unique_numbers[0]}).expand_dims(
            {DataKind.DAY_TYPE.value: [i, ]}) for i, a in enumerate(dataset.values) if
            (unique_numbers := np.unique(a, return_counts=True))], dim=DataKind.DAY_TYPE.value)


class Apply(Transform):
    _name = "apply"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.operation = kwargs.pop("operation", lambda x: x)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return self.operation(dataset)