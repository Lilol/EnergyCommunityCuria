from abc import abstractmethod
from typing import Iterable

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation.definitions import LoadMatchingMetric, ParametricEvaluationType
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import PhysicalParameterCalculator


class LoadMatchingParameterCalculator(PhysicalParameterCalculator):
    _parameter_calculated = LoadMatchingMetric.INVALID

    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        pass


class SelfConsumption(LoadMatchingParameterCalculator):
    _parameter_calculated = LoadMatchingMetric.SELF_CONSUMPTION

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        return input_da.sel(data=DataKind.P_SH).sum() / input_da.sel(data=DataKind.P_INJ).sum()


class SelfSufficiency(PhysicalParameterCalculator):
    _parameter_calculated = LoadMatchingMetric.SELF_SUFFICIENCY

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        return input_da.sel(data=DataKind.P_SH).sum() / input_da.sel(data=DataKind.P_WITH).sum()


class LoadMatchingMetricEvaluator(ParametricEvaluator):
    _type = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _name = "load_matching_metric_evaluation"
    _parameter_calculators = {LoadMatchingMetric.SELF_CONSUMPTION: SelfConsumption(),
                              LoadMatchingMetric.SELF_SUFFICIENCY: SelfSufficiency()}
