from pandas import DataFrame

from data_storage.data_store import DataStore
from input.definitions import DataKind
from output.write import Write
from parameteric_evaluation.definitions import ParametricEvaluationType
from parameteric_evaluation.run_parametric_evaluation import ParametricEvaluator
from parameteric_evaluation.physical import PhysicalParameterEvaluator
from visualization.processing_visualization import plot_shared_energy, plot_sci


def calculate_shared_energy(df, n_fam):
    calc_sum_consumption(df, n_fam)
    df[DataKind.SHARED] = df[[DataKind.PRODUCTION, DataKind.CONSUMPTION]].min(axis="rows")


def calc_sum_consumption(df, n_fam):
    df[DataKind.CONSUMPTION] = df[DataKind.CONSUMPTION_OF_FAMILIES] * n_fam + df[DataKind.CONSUMPTION_OF_USERS]


def calculate_sc(df):
    return df[DataKind.SHARED].sum() / df[DataKind.PRODUCTION].sum()


class TimeAggregationEvaluator(ParametricEvaluator):
    _type = ParametricEvaluationType.TIME_AGGREGATION

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
            results.loc[n_fam, 'sc_tou'] = PhysicalParameterEvaluator.calculate_theoretical_limit_of_self_consumption(
                tou_months, n_fam)
            for label, tr in time_resolution.items():
                results.loc[n_fam, label] = PhysicalParameterEvaluator.calculate_sc_for_time_aggregation(energy_year,
                                                                                                         tr, n_fam)
            calculate_shared_energy(energy_year, n_fam)
            energy_by_day = energy_year.groupby(time_resolution["sc_day"])
            plot_shared_energy(energy_by_day.sum()[DataKind.SHARED],
                               energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"), n_fam)
        Write().write(results, "self_consumption_for_various_time_aggregations")
        plot_sci(time_resolution, evaluation_parameters.number_of_families, results)
