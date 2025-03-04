from typing import Iterable

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation.definitions import Parameter


class Calculator:
    _name = "calculator"
    _parameter_calculated = Parameter
    calculators = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls.calculators) == 0:
            cls.calculators = {}
        cls.calculators[cls._parameter_calculated] = cls()

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        raise NotImplementedError("'calculate' method must be implemented individually in each Calculator class")

    def __call__(self, *args, **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        input_da = kwargs.pop("input_da", args[0])
        output = kwargs.pop("output", args[1])
        return self.calculate(input_da, output, **kwargs)


class MultiStepCalculation(PipelineStage):
    stage = Stage.ANALYZE
    _name = "evaluation"
    _calculators: dict[Calculator] = None

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        results = OmnesDataArray(dims=[DataKind.TIME.value, DataKind.METRIC.value],
                                 coords={DataKind.TIME.value: dataset[DataKind.TIME.value],
                                         DataKind.METRIC.value: list(self._calculators.keys())})
        for metric, calculator in self._calculators.items():
            calculator.calculate(dataset, results)
        return results
