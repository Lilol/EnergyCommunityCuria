import logging

import numpy as np

import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName

logger = logging.getLogger(__name__)


class DataExtractor(PipelineStage):
    stage = Stage.EXTRACT

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


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
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @staticmethod
    def set_value_and_check(value, section_in_config, key_in_config, setter=None, check=True):
        if check:
            value_cf = configuration.config.getint(section_in_config, key_in_config)
            if value != value_cf:
                logger.warning(
                    f"The value of [{section_in_config}.{key_in_config}] from input files ({value}) does not equal to "
                    f"the value set from the configuration file ({value_cf})")
        if setter is None:
            configuration.config.setint(section_in_config, key_in_config, value)
        else:
            getattr(configuration.config, setter)(section_in_config, key_in_config, value)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        # total number and list of tariff time-slots (index f)
        # ARERA's day-types depending on subdivision into tariff time-slots
        # NOTE : f : 1 - tariff time-slot F1, central hours of work-days
        #            2 - tariff time-slot F2, evening of work-days, and saturdays
        #            3 - tariff times-lot F2, night, sundays and holidays
        tariff_time_slots = dataset.unique()
        self.set_value_and_check(tariff_time_slots, "tariff", "tariff_time_slots", configuration.config.setarray,
                                 check=False)
        self.set_value_and_check(len(tariff_time_slots), "tariff", "number_of_time_of_use_periods")

        # time-steps where there is a change of tariff time-slot
        h_switch_arera = np.where(np.diff(np.insert(dataset.flatten(), -1, dataset[0, 0])) != 0)[0]
        self.set_value_and_check(h_switch_arera, "tariff", "tariff_period_switch_time_steps")

        # number of day-types (index j)
        # NOTE : j : 0 - work-days (monday-friday)
        #            1 - saturdays
        #            2 - sundays and holidays
        self.set_value_and_check(dataset.size(axis=0), "time", "number_of_day_types")

        # number of time-steps during each day (index i)
        self.set_value_and_check(dataset.size(axis=1), "time", "number_of_time_steps_per_day")

        # total number of time-steps (index h)
        self.set_value_and_check(dataset.size, "time", "total_number_of_time_steps")

        return dataset


class TypicalLoadProfileExtractor(DataExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        return OmnesDataArray(data=dataset.set_index([ColumnName.USER_TYPE, ColumnName.MONTH]))
