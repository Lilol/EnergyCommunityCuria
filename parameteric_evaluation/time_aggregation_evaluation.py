import logging
from typing import override

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.output.write import Write2DData
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, TimeAggregation
from parameteric_evaluation.load_matching_evaluation import SelfConsumption
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import SharedEnergy
from visualization.processing_visualization import plot_shared_energy, plot_sci

logger = logging.getLogger(__name__)


class TimeAggregationParameterCalculator(Calculator):
    _key = TimeAggregation.ARBITRARY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        """Evaluate self consumption with given temporal aggregation and number of families."""
        aggregated = input_da.groupby(f"time.{cls._key.value}").sum() if cls._key else input_da
        aggregated, _ = SharedEnergy.calculate(aggregated)
        return input_da, SelfConsumption.calculate(aggregated)[1]

aggregation_dimensions = {TimeAggregation.HOUR: f"time.{DataKind.HOUR.value}", TimeAggregation.MONTH: f"time.{DataKind.MONTH.value}",
    TimeAggregation.SEASON: f"time.{DataKind.SEASON.value}", TimeAggregation.YEAR: f"time.{DataKind.YEAR.value}", }

# Dynamically create calculator subclasses
for ta_key, dims in aggregation_dimensions.items():
    if ta_key in {TimeAggregation.INVALID, TimeAggregation.THEORETICAL_LIMIT, TimeAggregation.ARBITRARY}:
        continue

    class_name = f"{ta_key.name.capitalize()}Calculator"

    def make_calculator(dim, ta_key):
        class _Calc(TimeAggregationParameterCalculator):
            _key = ta_key

            @classmethod
            def calculate(cls, input_da: OmnesDataArray | None = None,
                          results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
                OmnesDataArray, float | None]:
                return super().calculate(input_da, results_of_previous_calculations, *args, **kwargs)
        _Calc.__name__ = class_name
        return _Calc

    globals()[class_name] = make_calculator(dims, ta_key)


class TheoreticalLimit(TimeAggregationParameterCalculator):
    _key = TimeAggregation.THEORETICAL_LIMIT

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        tou_months = DataStore()["tou_months"]
        SharedEnergy.calculate(tou_months)
        return input_da, SelfConsumption.calculate(tou_months)[1]


class TimeAggregationEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.TIME_AGGREGATION
    _name = "Time aggregation evaluator"

    @classmethod
    @override
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
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
