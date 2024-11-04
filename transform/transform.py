from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray


class DataTransformer(PipelineStage):
    stage = Stage.TRANSFORM

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        pass
