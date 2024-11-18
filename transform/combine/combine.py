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