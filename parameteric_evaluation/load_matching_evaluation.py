from abc import abstractmethod
from typing import Iterable

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import LoadMatchingMetric, ParametricEvaluationType, PhysicalMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class LoadMatchingParameterCalculator(Calculator):
    _key = LoadMatchingMetric.INVALID

    @classmethod
    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        pass


class SelfConsumption(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_CONSUMPTION
        
    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        return input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum() / input_da.sel(
            {DataKind.CALCULATED.value: PhysicalMetric.INJECTED_ENERGY}).sum()


class SelfSufficiency(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_SUFFICIENCY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        return input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum() / input_da.sel(
            {DataKind.CALCULATED.value: PhysicalMetric.WITHDRAWN_ENERGY}).sum()


class LoadMatchingMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _name = "load_matching_metric_evaluation"

