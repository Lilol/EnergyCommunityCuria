from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray


class Visualize(PipelineStage):
    stage = Stage.VISUALIZE
    _name = "visualize"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.vis_function = args[0]

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        # self.vis_function(dataset, *args, **kwargs)
        return dataset
