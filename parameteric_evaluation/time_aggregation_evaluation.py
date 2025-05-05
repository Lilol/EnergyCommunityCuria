from pandas import DataFrame

from data_storage.data_store import DataStore
from io_operation.input.definitions import DataKind
from io_operation.output.write import Write, Write2DData

from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric
from parameteric_evaluation.load_matching_evaluation import SelfConsumption
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import SharedEnergy
from visualization.processing_visualization import plot_shared_energy, plot_sci


def calculate_theoretical_limit_of_self_consumption(dataset):
    SharedEnergy.calculate(dataset)
    return SelfConsumption.calculate(dataset)


def calculate_sc_for_time_aggregation(dataset, time_resolution):
    """Evaluate self consumption with given temporal aggregation and number of families."""
    SharedEnergy.calculate(dataset.groupby(time_resolution).sum())
    return SelfConsumption.calculate(dataset)


class TimeAggregationEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.TIME_AGGREGATION

    @classmethod
    def invoke(cls, *args, **kwargs):
        evaluation_parameters = args[0]
        time_resolution = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH,
                               sc_week=DataKind.WEEK, sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                               sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])
        results = DataFrame(index=evaluation_parameters.number_of_families,
                            columns=list(time_resolution.keys()) + ["sc_tou"])
        results.index.name = "number_of_families"
        ds = DataStore()
        tou_months = ds["tou_months"]
        energy_year = ds["energy_year"]
        for n_fam in evaluation_parameters.number_of_families:
            results.loc[n_fam, 'sc_tou'] = calculate_theoretical_limit_of_self_consumption(tou_months)
            for label, tr in time_resolution.items():
                results.loc[n_fam, label] = calculate_sc_for_time_aggregation(energy_year, tr)
            SharedEnergy.calculate(energy_year)
            energy_by_day = energy_year.groupby(time_resolution["sc_day"])
            plot_shared_energy(energy_by_day.sum()[PhysicalMetric.SHARED_ENERGY],
                               energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"), n_fam)
        Write2DData().write(results, attribute="time_aggregation")
        plot_sci(time_resolution, evaluation_parameters.number_of_families, results)
