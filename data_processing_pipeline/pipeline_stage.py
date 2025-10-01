from typing import Optional

from data_processing_pipeline.definitions import Stage
from data_storage.omnes_data_array import OmnesDataArray


class PipelineStage:
    stage = Stage.INVALID
    _name = ""

    def __init__(self, name=_name, *args, **kwargs):
        if name is None:
            name = self._name
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def set_name(self, name):
        self.name = name
        return self

    def execute(self, dataset: Optional[OmnesDataArray], *args, **kwargs) -> Optional[OmnesDataArray]:
        raise NotImplementedError(
            "'execute' is not implemented in base class 'PipelineStage', 'execute"
            "' must be implemented in every child class.")

    def get_arg(self, name, **kwargs):
        fallback = kwargs.get('fallback', None)
        return kwargs.get(name, self.kwargs.get(name, fallback))
