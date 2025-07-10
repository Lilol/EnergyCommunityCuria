from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import LoadMatchingMetric, ParametricEvaluationType, PhysicalMetric, \
    OtherParameters
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class LoadMatchingParameterCalculator(Calculator):
    _key = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _nominator = PhysicalMetric.SHARED_ENERGY
    _denominator = OtherParameters.INVALID

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        value = input_da.sel({DataKind.CALCULATED.value: cls._nominator}).sum() / input_da.sel(
            {DataKind.CALCULATED.value: cls._denominator}).sum()
        return cls._postprocess_result(input_da, value)

    @classmethod
    def _postprocess_result(cls, input_da: OmnesDataArray | None, value):
        return input_da, value


class SelfConsumption(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_CONSUMPTION
    _denominator = OtherParameters.INJECTED_ENERGY


class SelfSufficiency(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_SUFFICIENCY
    _denominator = OtherParameters.WITHDRAWN_ENERGY


class SelfProduction(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.SELF_PRODUCTION
    _denominator = [OtherParameters.WITHDRAWN_ENERGY, OtherParameters.INJECTED_ENERGY]


class GridLiability(LoadMatchingParameterCalculator):
    _key = LoadMatchingMetric.GRID_LIABILITY
    _nominator = OtherParameters.INJECTED_ENERGY
    _denominator = OtherParameters.WITHDRAWN_ENERGY

    @classmethod
    def _postprocess_result(cls, input_da: OmnesDataArray | None, value):
        return input_da, value-1


class LoadMatchingMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.LOAD_MATCHING_METRICS
    _name = "load_matching_metric_evaluation"
