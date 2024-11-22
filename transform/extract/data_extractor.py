import logging

import numpy as np
import xarray as xr
from pandas import date_range, DataFrame

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName
from utility import configuration
from utility.day_of_the_week import get_weekday_code

logger = logging.getLogger(__name__)


class DataExtractor(PipelineStage):
    stage = Stage.EXTRACT
    _name = "data_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class TypicalYearExtractor(DataExtractor):
    _name = "typical_year_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dds = xr.concat([xr.concat([dd.assign_coords({ColumnName.TIME.value: dd.time.dt.hour}).rename(
            {ColumnName.TIME.value: ColumnName.HOUR.value}).expand_dims(ColumnName.DAY_OF_MONTH.value).assign_coords(
            {ColumnName.DAY_OF_MONTH.value: [day, ]}) for day, dd in df.groupby(df.time.dt.day)],
            dim=ColumnName.DAY_OF_MONTH.value).expand_dims(ColumnName.MONTH.value).assign_coords(
            {ColumnName.MONTH.value: [month, ]}) for month, df in dataset.groupby(dataset.time.dt.month)],
                        dim=ColumnName.MONTH.value)

        day_types = DataStore()["day_types"]
        return OmnesDataArray(xr.concat([
            dds.where(day_types.where(day_types == i)).mean(ColumnName.DAY_OF_MONTH.value, skipna=True).expand_dims(
                {ColumnName.DAY_TYPE.value: [i, ]}) for i in
            range(configuration.config.getint("time", "number_of_day_types"))], dim=ColumnName.DAY_TYPE.value))


class TariffExtractor(DataExtractor):
    def __init__(self, name="tariff_extractor", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    """
    ______
    NOTES
    Notation used in the variables
      - 'f' : tariff timeslot index, \in [1, nf] \subset N
      - 'j' : day-type index, \in [0, n_j) \subset N
      - 'i' : time step index during one day, \in [0, n_i) \subset N
      - 'h' : time step index in multiple days, \in [0, n_h) \subset N
    _____

    # total number and list of tariff time-slots (index f)
    # ARERA's day-types depending on subdivision into tariff time-slots
    # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
    #            2 - tariff time-slot F2, evening of work-days, and saturdays
    #            3 - tariff times-lot F2, night, sundays and holidays
    """

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        tariff_time_slots = np.unique(dataset)
        configuration.config.set_and_check("tariff", "tariff_time_slots", tariff_time_slots,
                                           configuration.config.setarray, check=False)
        configuration.config.set_and_check("tariff", "number_of_time_of_use_periods", len(tariff_time_slots))

        # time-steps where there is a change of tariff time-slot
        h_switch_arera = np.where(dataset[:, :-1].values - dataset[:, 1:].values)
        h_switch_arera = (h_switch_arera[0], h_switch_arera[1] + 1)
        configuration.config.set_and_check("tariff", "tariff_period_switch_time_steps", h_switch_arera,
                                           configuration.config.setarray, check=False)

        # number of day-types (index j)
        # NOTE : j : 0 - work-days (monday-friday)
        #            1 - saturdays
        #            2 - sundays and holidays
        configuration.config.set_and_check("time", "number_of_day_types", dataset.shape[0])

        # number of time-steps during each day (index i)
        configuration.config.set_and_check("time", "number_of_time_steps_per_day", dataset.shape[1])

        # total number of time-steps (index h)
        configuration.config.set_and_check("time", "total_number_of_time_steps", dataset.size)

        return dataset


class TouExtractor(DataExtractor):
    _name = "tou_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return xr.concat([OmnesDataArray(unique_numbers[1], dims=ColumnName.COUNT.value,
                                         coords={ColumnName.COUNT.value: unique_numbers[0]}).expand_dims(
            {ColumnName.DAY_TYPE.value: [i, ]}) for i, a in enumerate(dataset.values) if
            (unique_numbers := np.unique(a, return_counts=True))], dim=ColumnName.DAY_TYPE.value)


class DayTypeExtractor(DataExtractor):
    _name = "day_type_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        ref_year = configuration.config.get("time", "year")
        start = kwargs.pop("start", f"{ref_year}-01-01")
        end = kwargs.pop("end", f"{ref_year}-12-31")
        index = date_range(start=start, end=end, freq="d")
        ref_df = DataFrame(data=index.map(get_weekday_code), index=index, columns=[ColumnName.DAY_TYPE, ])
        return xr.concat([
            OmnesDataArray(df.astype(int).set_index(df.index.day).rename(columns={ColumnName.DAY_TYPE: month}),
                           dims=(ColumnName.DAY_OF_MONTH.value, ColumnName.MONTH.value)) for month, df in
            ref_df.groupby(ref_df.index.month)], dim=ColumnName.MONTH.value)


class DayCountExtractor(DataExtractor):
    _name = "day_count_extractor"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        dataset = xr.concat([OmnesDataArray(unique_numbers[1], dims=ColumnName.DAY_TYPE.value,
                                            coords={ColumnName.DAY_TYPE.value: unique_numbers[0]}).expand_dims(
            {ColumnName.MONTH.value: [i, ]}) for i, da in enumerate(dataset.T, 1) if
            (unique_numbers := np.unique(da, return_counts=True))], dim=ColumnName.MONTH.value).drop(
            dim=ColumnName.DAY_TYPE.value, labels=np.nan).fillna(0).astype(int)
        return dataset.assign_coords({ColumnName.DAY_TYPE.value: dataset[ColumnName.DAY_TYPE.value].astype(int)})
