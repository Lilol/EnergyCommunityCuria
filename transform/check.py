import logging

import pandas as pd
import xarray as xr

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from utility import configuration

logger = logging.getLogger(__name__)


class Check(PipelineStage):
    stage = Stage.CHECK
    _name = "data_transformer"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class CheckAnnualSum(Check):
    _name = "annual_sum_checker"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        time_of_use_labels = configuration.config.getarray("tariff", "time_of_use_labels", str)
        grouper = xr.DataArray(pd.MultiIndex.from_arrays(
            [dataset.sel(user_data=DataKind.MONTH).squeeze().values, dataset[DataKind.USER.value].values],
            names=[DataKind.MONTH.value, DataKind.USER.value], ), dims=DataKind.USER.value,
            coords={DataKind.USER.value: dataset[DataKind.USER.value].values}, )
        for (month, user), ds in dataset.groupby(grouper):
            time_of_use_consumption = ds.sel({DataKind.USER_DATA.value: time_of_use_labels}).sum()
            annual_consumption = ds.sel({DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).sum()
            if time_of_use_consumption.values != annual_consumption.values:
                logger.warning(
                    f"The sum of time of use energy consumption must match the aggregated consumption for every user and"
                    f" month, but discrepancy found for annual consumption: {annual_consumption.values} and summed tou "
                    f"consumption {time_of_use_consumption.values} for user {user} for month {month}.")
        dataset.loc[..., DataKind.ANNUAL_ENERGY] = dataset.loc[..., time_of_use_labels].sum(DataKind.USER_DATA.value)
        return dataset
