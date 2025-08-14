import logging
from typing import override

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.output.write import WriteDataArray
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, TimeAggregation, LoadMatchingMetric, \
    CombinedMetricEnum
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import SharedEnergy, PhysicalParameterCalculator
from visualization.processing_visualization import plot_shared_energy

logger = logging.getLogger(__name__)


class TimeAggregationParameterCalculator(Calculator):
    _key = None
    _metric = LoadMatchingMetric.INVALID
    _aggregation = TimeAggregation.INVALID

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        """Evaluate self consumption with given temporal aggregation and number of families."""
        aggregated = input_da.groupby(
            f"time.{cls._aggregation.value}").sum() if cls._aggregation != TimeAggregation.THEORETICAL_LIMIT else input_da
        aggregated, _ = SharedEnergy.calculate(aggregated)
        return input_da, PhysicalParameterCalculator.create(cls._metric).calculate(aggregated)[1]


# Dynamically create calculator subclasses
for ta_key in TimeAggregation:
    for metric in LoadMatchingMetric:
        if ta_key == TimeAggregation.INVALID or metric == LoadMatchingMetric.INVALID:
            continue

        class_name = f"{metric.value.replace(' ', '')}By{ta_key.name.title()}Calculator"


        def make_calculator(k, m):
            class _Calc(TimeAggregationParameterCalculator):
                _key = CombinedMetricEnum.from_parts(k, m)
                _metric = m
                _param_calculator = PhysicalParameterCalculator.create(_metric)
                _aggregation = k

                @classmethod
                def calculate(cls, input_da: OmnesDataArray | None = None,
                              results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
                    OmnesDataArray, float | None]:
                    """Evaluate load matching metric with given temporal aggregation."""
                    if cls._aggregation == TimeAggregation.HOUR:
                        aggregated = input_da.groupby(
                            ((input_da.time.dt.dayofyear - 1) * 24 + input_da.time.dt.hour)).mean()
                    else:
                        aggregated = input_da.groupby(
                            f"time.{cls._aggregation.value}").sum() if cls._aggregation != TimeAggregation.THEORETICAL_LIMIT else input_da
                    aggregated, _ = SharedEnergy.calculate(aggregated)
                    return input_da, PhysicalParameterCalculator.create(cls._metric).calculate(aggregated)[1]

            _Calc.__name__ = class_name
            return _Calc


        globals()[class_name] = make_calculator(ta_key, metric)


class TheoreticalLimit(TimeAggregationParameterCalculator):
    _key = TimeAggregation.THEORETICAL_LIMIT


class TimeAggregationEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.TIME_AGGREGATION
    _name = "Time aggregation evaluator"

    @staticmethod
    def get_eval_metrics(evaluation_type):
        return {CombinedMetricEnum.from_parts(key, m): TimeAggregationParameterCalculator.get_subclass(
            CombinedMetricEnum.from_parts(key, m)) for key in TimeAggregation for m in LoadMatchingMetric if
            m != LoadMatchingMetric.INVALID and key != TimeAggregation.INVALID}

    @classmethod
    @override
    def invoke(cls, *args, **kwargs):
        input_da, results = super().invoke(*args, **kwargs)
        plot_shared_energy(input_da, args[2]["number_of_families"], args[2]["battery_size"])
        WriteDataArray().execute(results, attribute="time_aggregation")
        return input_da, results
