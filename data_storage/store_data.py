from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray


class Store(PipelineStage):
    stage = Stage.STORE
    _name = "store_data"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.key = kwargs.pop("key", name)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        DataStore()[self.key] = dataset
        return dataset
