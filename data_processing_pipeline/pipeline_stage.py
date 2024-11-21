from data_processing_pipeline.definitions import Stage
from data_storage.dataset import OmnesDataArray


class PipelineStage:
    stage = Stage.INVALID
    _name = ""

    def __init__(self, name=_name, *args, **kwargs):
        self.name = name

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        raise NotImplementedError(
            "'execute' is not implemented in base class 'PipelineStage', must be implemented in every child class.")
