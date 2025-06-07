from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.omnes_data_array import OmnesDataArray


class CollectPipelineStage(PipelineStage, DataProcessingPipeline):
    stage = Stage.OTHER
    _name = "collector_of_pipeline_stages"

    def __init__(self, name=_name, *args, **kwargs):
        super(DataProcessingPipeline).__init__(name, *args, **kwargs)
        super(PipelineStage).__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        return super(DataProcessingPipeline).execute(*args, **kwargs, dataset=dataset)

