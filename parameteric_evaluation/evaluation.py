from typing import Iterable

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from parameteric_evaluation.definitions import Parameter


class Calculator:
    _name = "calculator"
    _parameter_calculated = Parameter.INVALID
    calculators = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls.calculators) == 0:
            cls.calculators = {}
        cls.calculators[cls._parameter_calculated] = cls()

    def calculate(self, input_da: OmnesDataArray, output: OmnesDataArray, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        pass


class Evaluate(PipelineStage):
    stage = Stage.ANALYZE
    _name = "evaluation"
    _calculators: Iterable[Calculator] = None

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        pass
