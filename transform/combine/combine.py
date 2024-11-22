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
    # Class to evaluate monthly consumption from hourly load profiles
    # evaluate the monthly consumption divided into tariff time-slots from the
    # hourly load profiles in the day-types
    _name = 'typical_monthly_consumption_calculator'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        data_store = DataStore()
        time_of_use_time_slots = data_store["time_of_use_time_slots"]
        day_count = data_store["day_count"]

        return OmnesDataArray(xr.concat([(dataset.isel({ColumnName.DAY_TYPE.value: dt,
                                                        ColumnName.HOUR.value: time_of_use_time_slots.sel(
                                                            {ColumnName.DAY_TYPE.value: dt}) == tou}).sum(
            ColumnName.HOUR.value) * day_count.sel({ColumnName.DAY_TYPE.value: dt})).expand_dims(
            (ColumnName.TARIFF_TIME_SLOT.value, ColumnName.DAY_TYPE.value)).assign_coords(
            {ColumnName.TARIFF_TIME_SLOT.value: [tou, ], ColumnName.DAY_TYPE.value: [dt, ]}) for tou in
            configuration.config.getarray("tariff", "tariff_time_slots", int) for dt in
            configuration.config.getarray("time", "day_types", int)], dim=ColumnName.DAY_TYPE.value).sum("day_type"))
