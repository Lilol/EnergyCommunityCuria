import logging

from pandas import DataFrame

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
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
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        """Evaluate self consumption with given temporal aggregation and number of families."""
        SharedEnergy.calculate(input_da.groupby(cls._key).sum())
        return input_da, SelfConsumption.calculate(input_da)[1]


class TheoreticalLimit(TimeAggregationParameterCalculator):
    _key = TimeAggregation.THEORETICAL_LIMIT

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        tou_months = DataStore()["tou_months"]
        SharedEnergy.calculate(tou_months)
        return input_da, SelfConsumption.calculate(tou_months)[1]


class TimeAggregationEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.TIME_AGGREGATION
    _name = "Time aggregation evaluator"

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        time_resolution = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH,
                               sc_week=DataKind.WEEK, sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                               sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])
        input_da, results = super().invoke(*args, **kwargs)
        plot_shared_energy(energy_by_day.sum()[PhysicalMetric.SHARED_ENERGY],
                           energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"),
                           args[2].number_of_families)
        Write2DData().write(results, attribute="time_aggregation")
        plot_sci(time_resolution, args[2].number_of_families, results)
        return input_da, results
