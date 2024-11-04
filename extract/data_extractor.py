import logging

import numpy as np

import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName, BillType, UserType

logger = logging.getLogger(__name__)


class DataExtractor(PipelineStage):
    stage = Stage.EXTRACT

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class UserExtractor(DataExtractor):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        dataset[ColumnName.USER_TYPE] = dataset[ColumnName.USER_TYPE].apply(lambda x: UserType(x))
        return dataset


class BillsExtractor(DataExtractor):
    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        user_data = args[0]

        def get_bill_type(df):
            return BillType.MONO if df[ColumnName.MONO_TARIFF].notna().all(how="all") else BillType.TIME_OF_USE

        dataset[ColumnName.BILL_TYPE] = user_data.groupby(ColumnName.USER).apply(get_bill_type)
        return dataset


class TariffExtractor(DataExtractor):
    """
    ______
    NOTES
    Notation used in the variables
      - 'f' : tariff timeslot index, \in [1, nf] \subset N
      - 'j' : day-type index, \in [0, n_j) \subset N
      - 'i' : time step index during one day, \in [0, n_i) \subset N
      - 'h' : time step index in multiple days, \in [0, n_h) \subset N
    _____
    """

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        # total number and list of tariff time-slots (index f)
        # ARERA's day-types depending on subdivision into tariff time-slots
        # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
        #            2 - tariff time-slot F2, evening of work-days, and saturdays
        #            3 - tariff times-lot F2, night, sundays and holidays
        tariff_time_slots = dataset.unique()
        configuration.config.set_and_check("tariff", "tariff_time_slots", tariff_time_slots,
                                           configuration.config.setarray, check=False)
        configuration.config.set_value_and_check("tariff", "number_of_time_of_use_periods", len(tariff_time_slots))

        # time-steps where there is a change of tariff time-slot
        h_switch_arera = np.where(np.diff(np.insert(dataset.flatten(), -1, dataset[0, 0])) != 0)[0]
        configuration.config.set_and_check("tariff", "tariff_period_switch_time_steps", h_switch_arera)

        # number of day-types (index j)
        # NOTE : j : 0 - work-days (monday-friday)
        #            1 - saturdays
        #            2 - sundays and holidays
        configuration.config.set_and_check("time", "number_of_day_types", dataset.size(axis=0))

        # number of time-steps during each day (index i)
        configuration.config.set_and_check("time", "number_of_time_steps_per_day", dataset.size(axis=1))

        # total number of time-steps (index h)
        configuration.config.set_and_check("time", "total_number_of_time_steps", dataset.size)

        return dataset


class TypicalLoadProfileExtractor(DataExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        return OmnesDataArray(data=dataset.set_index([ColumnName.USER_TYPE, ColumnName.MONTH]))
