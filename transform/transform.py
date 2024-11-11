import numpy as np
import xarray as xr
from pandas import DataFrame, timedelta_range, date_range

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, UserType, BillType
from utility import configuration
from utility.day_of_the_week import df_year, cols_to_add


class DataTransformer(PipelineStage):
    stage = Stage.TRANSFORM

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class UserDataTransformer(DataTransformer):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        dataset[ColumnName.USER_TYPE] = dataset[ColumnName.USER_TYPE].apply(lambda x: UserType(x))
        return dataset


class BillDataTransformer(DataTransformer):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        user_data = args[0]

        def get_bill_type(df):
            return BillType.MONO if df[ColumnName.MONO_TARIFF].notna().all(how="all") else BillType.TIME_OF_USE

        dataset[ColumnName.BILL_TYPE] = user_data.groupby(ColumnName.USER).apply(get_bill_type)
        return dataset


class ReshaperByYear(DataTransformer):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        user_name = kwargs.pop('user_name')
        ref_year = configuration.config.getint("time", "year")
        profile = OmnesDataArray(data=DataFrame(data=dataset, columns=timedelta_range(start="0 Days",
                                                                                      freq=configuration.config.get(
                                                                                          "time", "resolution"),
                                                                                      periods=dataset.shape[1]),
                                                index=date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31",
                                                                 freq="d")))
        profile[ColumnName.USER] = user_name
        profile[cols_to_add] = df_year[cols_to_add]
        return profile


class TypicalLoadProfileTransformer(DataTransformer):
    def __init__(self, name="typical_load_profile_transformer", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset.loc[:, ColumnName.USER_TYPE] = xr.apply_ufunc(lambda x: UserType(x),
                                                              dataset.sel({"dim_1": ColumnName.USER_TYPE}),
                                                              vectorize=True)
        values = dataset.where(~dataset.dim_1.isin((ColumnName.USER_TYPE, ColumnName.MONTH)), drop=True)
        tariff_time_slots = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[1].strip("j")) + 1,
                                           values.dim_1, vectorize=True)
        hour = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), values.dim_1,
                              vectorize=True)
        values.reset_index(dims_or_levels=("dim_0", "dim_1"))

        dims = (ColumnName.TARIFF_TIME_SLOT.value, ColumnName.USER_TYPE.value, ColumnName.HOUR.value,
                ColumnName.MONTH.value)
        coords = {ColumnName.TARIFF_TIME_SLOT.value: np.unique(tariff_time_slots),
                  ColumnName.HOUR.value: np.unique(hour),
                      ColumnName.USER_TYPE.value: np.unique(dataset.loc[:, ColumnName.USER_TYPE]),
                  ColumnName.MONTH.value: np.unique(dataset.loc[:, ColumnName.MONTH])}
        new_array = xr.DataArray(dims=dims, coords=coords)
        for user_type, df in values.groupby(dataset.loc[:, ColumnName.USER_TYPE]):
            hours = xr.apply_ufunc(lambda x: x if type(x) != str else int(x.split("_")[2].strip("i")), df.dim_1,
                                   vectorize=True)
            for hour, df in df.groupby(hours):
                new_array.loc[:, user_type, hour, :]=df.values.T
        return values


class PvPlantDataTransformer(DataTransformer):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        # Add column with yearly production by ToU tariff for each plant
        dataset[ColumnName.MONO_TARIFF] = np.nan
        data_plants_tou = kwargs.pop("data_plants_tou")
        tou_columns = configuration.config.getarray("tariff", "time_of_use_labels")
        dataset = dataset.merge(
            data_plants_tou.groupby(ColumnName.USER)[tou_columns].sum().rename(tou_columns).reset_index(),
            on=ColumnName.USER)

        # Add column with type of plant
        dataset[ColumnName.USER_TYPE] = UserType.PV if ColumnName.USER_TYPE not in dataset else dataset[
            ColumnName.USER_TYPE].fillna(UserType.PV)

        return dataset


class TariffTransformer(DataTransformer):
    def __init__(self, name="tariff_transformer", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        # Tariff timeslot naming convention: time slot indices start from 0
        dataset = dataset.rename({"dim_0": ColumnName.DAY_TYPE.value, "dim_1": ColumnName.HOUR.value})
        dataset = dataset - 1
        dataset[ColumnName.DAY_TYPE.value] = dataset[ColumnName.DAY_TYPE.value].astype(int) + 1
        dataset[ColumnName.HOUR.value] = dataset[ColumnName.HOUR.value].astype(int)
        return dataset
