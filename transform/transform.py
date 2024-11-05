import numpy as np
from pandas import DataFrame, timedelta_range, date_range

import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, UserType, BillType
from time.day_of_the_week import df_year, cols_to_add


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
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        return OmnesDataArray(data=dataset.set_index([ColumnName.USER_TYPE, ColumnName.MONTH]))


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
