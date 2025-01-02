import numpy as np
from pandas import DataFrame

from data_storage.data_store import DataStore, logger
from input.definitions import DataKind
from output.write import Write
from parameteric_evaluation.battery_management import manage_bess
from parameteric_evaluation.dataset_creation import create_dataset_for_parametric_evaluation
from parameteric_evaluation.definitions import ParametricEvaluationType
from parameteric_evaluation.economic import eval_capex
from parameteric_evaluation.environmental import eval_co2
from parameteric_evaluation.physical import calculate_theoretical_limit_of_self_consumption, \
    calculate_sc_for_time_aggregation, eval_physical_parameters
from parameteric_evaluation.target_self_consumption import find_the_optimal_number_of_families_for_sc_ratio, \
    calculate_shared_energy
from utility import configuration
from utility.dimensions import power_to_energy
from visualization.processing_visualization import plot_sci, plot_shared_energy

ref_year = configuration.config.getint("time", "year")

# Evaluate the number of families to reach set targets
create_dataset_for_parametric_evaluation()
ds = DataStore()
tou_months, energy_year = ds["tou_months"], ds["energy_year"]

evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")


def define_output_df(evaluation_types, scenarios):
    physical_metrics = ["sc", "ss", "e_sh", "e_inj",
                        "e_with"] if ParametricEvaluationType.PHYSICAL_METRICS in evaluation_types else []
    environmental_metrics = ["esr", "em_tot",
                             "em_base"] if ParametricEvaluationType.ENVIRONMENTAL_METRICS in evaluation_types else []
    economic_metrics = ["capex", ] if ParametricEvaluationType.ECONOMIC_METRICS in evaluation_types else []
    return DataFrame(index=scenarios.index, columns=physical_metrics + environmental_metrics + economic_metrics)


def calculate_metrics():
    scenarios = DataFrame(data=parameters.combinations, columns=[DataKind.NUMBER_OF_FAMILIES, DataKind.BATTERY_SIZE])
    # Get plants sizes and number of users
    n_users = len(ds["data_users"])
    data_plants = ds["data_plants"]
    pv_sizes = list(data_plants.loc[data_plants[DataKind.USER_TYPE] == 'pv', DataKind.POWER])
    if len(pv_sizes) < len(data_plants):
        raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")
    # Initialize results
    results = define_output_df(evaluation_types, scenarios)
    p_prod = energy_year.sel(user=DataKind.PRODUCTION)
    e_prod = power_to_energy(p_prod)
    p_cons = energy_year.sel(user=DataKind.CONSUMPTION_OF_USERS)
    p_fam = energy_year.sel(user=DataKind.CONSUMPTION_OF_FAMILIES)
    # Evaluate each scenario
    for i, scenario in scenarios.iterrows():
        # Get configuration
        n_fam = scenario[DataKind.NUMBER_OF_FAMILIES]
        bess_size = scenario[DataKind.BATTERY_SIZE]

        # Calculate withdrawn power
        p_with = p_cons + n_fam * p_fam
        # Manage BESS, if present
        p_inj = p_prod - manage_bess(p_prod, p_with, bess_size)

        # Eval REC
        e_cons = power_to_energy(p_with)
        if ParametricEvaluationType.PHYSICAL_METRICS in evaluation_types:
            results.loc[i, ["sc", "ss", "e_sh", "e_inj", "e_with"]] = eval_physical_parameters(p_inj, p_with)

        # Evaluate emissions
        if ParametricEvaluationType.ENVIRONMENTAL_METRICS in evaluation_types:
            results.loc[i, ["esr", "em_tot", "em_base"]] = eval_co2(results.loc[i, "e_sh"],
                                                                    e_cons=results.loc[i, "e_with"],
                                                                    e_inj=results.loc[i, "e_inj"], e_prod=e_prod)

        # Evaluate CAPEX
        if ParametricEvaluationType.ECONOMIC_METRICS in evaluation_types:
            results.loc[i, "capex"] = eval_capex(pv_sizes, bess_size=bess_size, n_users=n_users + n_fam)
    Write().write(results, "results")


def calculate_self_consumption_for_various_time_aggregations(evaluation_parameters):
    time_resolution = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH,
                           sc_week=DataKind.WEEK, sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                           sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])
    results = DataFrame(index=evaluation_parameters.number_of_families,
                        columns=list(time_resolution.keys()) + ["sc_tou"])
    results.index.name = "number_of_families"
    for n_fam in evaluation_parameters.number_of_families:
        results.loc[n_fam, 'sc_tou'] = calculate_theoretical_limit_of_self_consumption(tou_months, n_fam)
        for label, tr in time_resolution.items():
            results.loc[n_fam, label] = calculate_sc_for_time_aggregation(energy_year, tr, n_fam)
        calculate_shared_energy(energy_year, n_fam)
        energy_by_day = energy_year.groupby(time_resolution["sc_day"])
        plot_shared_energy(energy_by_day.sum()[DataKind.SHARED],
                           energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"), n_fam)
    Write().write(results, "self_consumption_for_various_time_aggregations")
    plot_sci(time_resolution, evaluation_parameters.number_of_families, results)


def evaluate_self_consumption_targets():
    self_consumption_targets = configuration.config.getarray("parametric_evaluation", "self_consumption_targets")
    max_number_of_households = configuration.config.getint("parametric_evaluation", "max_number_of_households")
    df = DataFrame(np.nan, index=self_consumption_targets, columns=["number_of_families", "self_consumption_realized"])
    # Evaluate number of families for each target
    sc, nf = 0, 0
    for sc_target in self_consumption_targets:
        # # Skip if previous target was already higher than this
        if sc >= sc_target:
            df.loc[sc_target, ["number_of_families", "self_consumption_realized"]] = nf, sc
            continue

        # # Find number of families to reach target
        nf, sc = find_the_optimal_number_of_families_for_sc_ratio(energy_year, sc_target, max_number_of_households)

        # Update
        df.loc[sc_target, ["number_of_families", "self_consumption_realized"]] = nf, sc

        # # Exit if targets cannot be reached
        if sc < sc_target:
            logger.warning("Exiting loop because requirement cannot be reached.")
            break
        if nf >= max_number_of_households:
            logger.warning("Exiting loop because max families was reached.")
            break
    logger.info(
        f"Self consumption targets reached; number of families:\n{','.join(f'{row.number_of_families}; {row.self_consumption_realized}' for _, row in df.iterrows())}")
    logger.info(f"To provide battery sizes for evaluation in the configuration file use:\n[parametric_evaluation]"
                f"\nevaluation_parameters={{'bess_sizes': [...], 'number_of_families': [{f','.join(f'{nf}' for nf in df.number_of_families)}]}}\nor\n"
                f"evaluation_parameters = {{'number_of_families': {f','.join(f'{nf}: [...]' for nf in df.number_of_families)}}}")
    logger.info(f"For battery evaluation please set \n[parametric_evaluation]\nuse_bess=True")
    return df


# n_fams can come from this or just from the configuration file
# if n_fams come from here, set the parametric evaluation object's family sizes
if ParametricEvaluationType.SELF_CONSUMPTION_TARGETS in evaluation_types:
    evaluate_self_consumption_targets()

# ----------------------------------------------------------------------------
# Setup to aggregate in time
# Function to evaluate SELF_CONSUMPTION with different aggregations in time
parameters = configuration.config.get("parametric_evaluation", "evaluation_parameters")

if ParametricEvaluationType.SELF_CONSUMPTION_FOR_TIME_AGGREGATION in evaluation_types:
    calculate_self_consumption_for_various_time_aggregations(parameters)

# ----------------------------------------------------------------------------
# Calculate various metrics
if not any(metr in evaluation_types for metr in (
        ParametricEvaluationType.PHYSICAL_METRICS, ParametricEvaluationType.ENVIRONMENTAL_METRICS,
        ParametricEvaluationType.ECONOMIC_METRICS)):
    exit(0)

calculate_metrics()
