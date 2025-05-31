from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import LoadMatchingMetric, ParametricEvaluationType, PhysicalMetric, \
    OtherParameters
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class LoadMatchingParameterCalculator(Calculator):
    _key = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _relative_to = OtherParameters.INVALID

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        value = input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum() / input_da.sel(
            {DataKind.CALCULATED.value: cls._relative_to}).sum()
        return input_da, value


class SelfConsumption(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_CONSUMPTION
    _relative_to = OtherParameters.INJECTED_ENERGY


class SelfSufficiency(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_SUFFICIENCY
    _relative_to = OtherParameters.WITHDRAWN_ENERGY


class LoadMatchingMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _name = "load_matching_metric_evaluation"
