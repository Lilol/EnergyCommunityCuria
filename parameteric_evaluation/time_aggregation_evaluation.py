import logging
from typing import override

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.output.write import Write2DData
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, TimeAggregation, \
    LoadMatchingMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import SharedEnergy, PhysicalParameterCalculator
from visualization.processing_visualization import plot_shared_energy, plot_sci

logger = logging.getLogger(__name__)


class CombinedMetric:
    def __init__(self, first, second):
        assert hasattr(first, "value") and hasattr(second, "value") and hasattr(first, "name") and hasattr(second,
                                                                                                           "name")
        self.first = first
        self.second = second

    @property
    def value(self):
        return self.first.value, self.second.value

    @property
    def name(self):
        return f"{self.first.name}, {self.second.name}"

    def valid(self):
        return self.first.valid() and self.second.valid()

    def __eq__(self, other):
        if not isinstance(other, CombinedMetric):
            return False
        return (self.first, self.second) == (other.first, other.second)

    def __hash__(self):
        return hash((self.first, self.second))

    def __repr__(self):
        return f"(first={self.first.value!r}, second={self.second.value!r})"


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
                _key = CombinedMetric(k, m)
                _metric = m
                _param_calculator = PhysicalParameterCalculator.create(_metric)
                _aggregation = k

                @classmethod
                def calculate(cls, input_da: OmnesDataArray | None = None,
                              results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
                    OmnesDataArray, float | None]:
                    """Evaluate load matching metric with given temporal aggregation."""
                    aggregated = input_da.groupby(
                        f"time.{cls._aggregation.value}").sum() if cls._aggregation != TimeAggregation.THEORETICAL_LIMIT else \
                        DataStore()["tou_months"]
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
        return {CombinedMetric(key, m): TimeAggregationParameterCalculator.get_subclass(CombinedMetric(key, m)) for key
                in TimeAggregation for m in LoadMatchingMetric if
                m != LoadMatchingMetric.INVALID and key != TimeAggregation.INVALID}

    @classmethod
    @override
    def invoke(cls, *args, **kwargs):
        time_resolution = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH,
                               sc_week=DataKind.WEEK, sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                               sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])
        input_da, results = super().invoke(*args, **kwargs)
        energy_by_day = DataStore()["energy_by_year"].sum("day")
        plot_shared_energy(energy_by_day.sum()[PhysicalMetric.SHARED_ENERGY],
                           energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"),
                           args[2].number_of_families)
        Write2DData().write(results, attribute="time_aggregation")
        plot_sci(time_resolution, args[2].number_of_families, results)
        return input_da, results
