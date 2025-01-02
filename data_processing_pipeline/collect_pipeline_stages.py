from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray


class CollectPipelineStage(PipelineStage, DataProcessingPipeline):
    stage = Stage.OTHER
    _name = "collector_of_pipeline_stages"

    def __init__(self, name=_name, *args, **kwargs):
        super(DataProcessingPipeline).__init__(name, *args, **kwargs)
        super(PipelineStage).__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        return super(DataProcessingPipeline).execute(*args, **kwargs, dataset=dataset)

