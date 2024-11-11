import logging

import numpy as np

from utility import configuration
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray

logger = logging.getLogger(__name__)


class DataExtractor(PipelineStage):
    stage = Stage.EXTRACT

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


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

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
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
