import logging
from typing import Iterable

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import Parameter, PhysicalMetric
from utility.subclass_registration_base import SubclassRegistrationBase

logger = logging.getLogger(__name__)


def auto_hook_classmethod(method):
    @classmethod
    def wrapper(cls, *args, **kwargs):
        cls._print_name()  # hook: run before actual method
        return method.__func__(cls, *args, **kwargs)  # unwrap classmethod

    return wrapper


class Calculator(SubclassRegistrationBase):
    _key = PhysicalMetric
    _name = "calculator"

    @classmethod
    def _print_name(cls):
        # Handle None case for _key
        if cls._key is None:
            logger.info(f"Invoke calculator '{cls._name}'")
        elif isinstance(cls._key, tuple):
            logger.info(f"Invoke calculator for '{(cls._key[0].value, cls._key[1].value)}'")
        else:
            logger.info(f"Invoke calculator for '{cls._key.value}'")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Hook only if 'calculate' is defined in the subclass
        if 'calculate' in cls.__dict__:
            orig = cls.__dict__['calculate']
            if isinstance(orig, classmethod):
                wrapped = auto_hook_classmethod(orig)
                setattr(cls, 'calculate', wrapped)

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        raise NotImplementedError("'calculate' method must be implemented individually in each Calculator class")

    @classmethod
    def postprocess(cls, result, results_of_previous_calculations: OmnesDataArray | None, parameters: dict):
        """
        Unified output updater called after every calculation.
        Subclasses don't need to override this unless necessary.
        """
        if results_of_previous_calculations is None:
            return None

        if result is None:
            return results_of_previous_calculations

        if isinstance(result, tuple):
            _, data = result
        else:
            data = result
        return results_of_previous_calculations.update(data, {DataKind.METRIC.value: cls._key, **parameters})

    @classmethod
    def call(cls, input_da: OmnesDataArray | None = None,
             results_of_previous_calculations: OmnesDataArray | None = None, parameters: dict | None = None,
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
