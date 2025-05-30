from typing import Iterable

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import Parameter
from utility.subclass_registration_base import SubclassRegistrationBase


class Calculator(SubclassRegistrationBase):
    _key = Parameter
    _name = "calculator"

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        raise NotImplementedError("'calculate' method must be implemented individually in each Calculator class")

    @classmethod
    def postprocess(cls, result, results_of_previous_calculations: OmnesDataArray | None, parameters: dict):
        """
        Unified output updater called after every calculation.
        Subclasses don't need to override this unless necessary.
        """
        if results_of_previous_calculations is None or result is None:
            return result

        if isinstance(result, tuple):
            _, data = result
        else:
            data = result
        results_of_previous_calculations.update(data, {DataKind.METRIC.value: cls._key, **parameters})
        return results_of_previous_calculations

    @classmethod
    def call(cls, input_da: OmnesDataArray | None = None,
             results_of_previous_calculations: OmnesDataArray | None = None,
             parameters: dict | None = None,
             **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        input_da, result = cls.calculate(input_da, results_of_previous_calculations, **kwargs)
        return input_da, cls.postprocess(result, results_of_previous_calculations, parameters)


class MultiStepCalculation(PipelineStage):
    stage = Stage.ANALYZE
    _name = "evaluation"
    _calculators: dict[Calculator] = None

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        results = OmnesDataArray(dims=[DataKind.TIME.value, DataKind.METRIC.value],
                                 coords={DataKind.TIME.value: dataset[DataKind.TIME.value],
                                         DataKind.METRIC.value: list(self._calculators.keys())})
        for metric, calculator in self._calculators.items():
            calculator.calculate(dataset, results)
        return results
