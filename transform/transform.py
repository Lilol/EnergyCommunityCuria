import numpy as np
from pandas import DataFrame, timedelta_range, date_range, MultiIndex

import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, UserType, BillType
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
        dataset[ColumnName.USER_TYPE] = dataset[ColumnName.USER_TYPE].apply(lambda x: UserType(x))
        df = dataset.filter(regex='y.*', axis=1)
        tariff_time_slots = df.columns.str.split("_").str[1].str.strip("j").astype(int) + 1
        hours = df.columns.str.split("_").str[2].str.strip("i").astype(int)
        dataset = dataset.set_index((ColumnName.USER_TYPE, ColumnName.MONTH))
        df.columns = MultiIndex.from_tuples((i, j) for i, j in zip(tariff_time_slots, hours))
        df.columns.names = (ColumnName.TARIFF_TIME_SLOT, ColumnName.HOUR)
        df.index = dataset.index
        df.index.names = (x.value for x in df.index.names)
        df.columns.names = (x.value for x in df.columns.names)
        return OmnesDataArray.from_series(df.stack().stack())


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
        dataset = dataset - 1
        dataset[ColumnName.DAY_TYPE.value] = dataset[ColumnName.DAY_TYPE.value].astype(int) + 1
        dataset[ColumnName.HOUR.value] = dataset[ColumnName.HOUR.value].astype(int)
        return dataset
