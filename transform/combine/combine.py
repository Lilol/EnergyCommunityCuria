import xarray as xr

from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from utility import configuration


class Combine(PipelineStage):
    _name = 'combine'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError("'execute()' is not implemented in Combine base class.")


class CalculateTypicalMonthlyConsumption(Combine):
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

        return OmnesDataArray(xr.concat([(dataset.isel({DataKind.DAY_TYPE.value: dt,
                                                        DataKind.HOUR.value: time_of_use_time_slots.sel(
                                                            {DataKind.DAY_TYPE.value: dt}) == tou}).sum(
            DataKind.HOUR.value) * day_count.sel({DataKind.DAY_TYPE.value: dt})).expand_dims(
            (DataKind.TARIFF_TIME_SLOT.value, DataKind.DAY_TYPE.value)).assign_coords(
            {DataKind.TARIFF_TIME_SLOT.value: [tou, ], DataKind.DAY_TYPE.value: [dt, ]}) for tou in
            configuration.config.getarray("tariff", "tariff_time_slots", int) for dt in
            configuration.config.getarray("time", "day_types", int)], dim=DataKind.DAY_TYPE.value).sum(
            DataKind.DAY_TYPE.value))


class AddYearlyConsumptionToBillData(Combine):
    _name = 'yearly_consumption_combiner'

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        data_bills = DataStore()["bills"]
        dd = xr.concat([ds.sel({DataKind.USER_DATA.value: [DataKind.MONO_TARIFF,
                                                           *configuration.config.getarray("tariff",
                                                                                            "time_of_use_labels", str),
                                                           DataKind.ANNUAL_ENERGY]}).sum(
            dim=DataKind.USER.value).assign_coords({DataKind.USER.value: u}) for u, ds in
                        data_bills.groupby(DataKind.USER.value)], dim=DataKind.USER.value).astype(float)
        return xr.concat([dd, dataset.T], dim=DataKind.USER_DATA.value)
