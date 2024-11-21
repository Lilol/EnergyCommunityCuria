import xarray as xr

from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName
from utility import configuration


class Combine(PipelineStage):
    _name = 'combine'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        data_plants_tou = DataStore()["time_of_use_tariff"]
        tou_columns = configuration.config.getarray("tariff", "time_of_use_labels")
        dataset = dataset.merge(
            data_plants_tou.groupby(ColumnName.USER)[tou_columns].sum().rename(tou_columns).reset_index(),
            on=ColumnName.USER)
        return dataset


class TypicalMonthlyConsumptionCalculator(Combine):
    _name = 'typical_monthly_consumption_calculator'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        data_store = DataStore()
        typical_load_profile = data_store["typical_load_profile_gse"]
        time_of_use_time_slots = data_store["time_of_use_time_slots"]
        day_count = data_store["day_count"]

        dataset = OmnesDataArray(xr.concat([typical_load_profile.isel({ColumnName.DAY_TYPE.value: dt}, {
            ColumnName.HOUR.value: time_of_use_time_slots.sel({ColumnName.DAY_TYPE.value: 1}) == tou}).sum(
            ColumnName.HOUR.value) * day_count.sel({ColumnName.DAY_TYPE.value: dt}) for tou in
                                            configuration.config.getarray("tariff", "tariff_time_slots", int) for dt in
                                            range(configuration.config.getint("time", "number_of_day_types"))],
                                           dim=ColumnName.DAY_TYPE.value).groupby(ColumnName.DAY_TYPE.value).sum())
        return dataset
