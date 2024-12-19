import numpy as np
from pandas import DataFrame, to_datetime

from analysis.evaluation import eval_co2, eval_capex, energy, eval_physical_parameters, manage_bess, \
    find_the_optimal_number_of_families_for_sc_ratio, calculate_shared_energy, \
    calculate_theoretical_limit_of_self_consumption, calculate_sc_for_time_aggregation
from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from input.definitions import DataKind
from input.read import Read
from output.write import Write
from transform.combine.combine import ArrayConcat
from transform.transform import TransformCoordinateIntoDimension, Aggregate, Apply, Rename
from utility import configuration
from visualization.processing_visualization import plot_sci, plot_shared_energy

ref_year = configuration.config.getint("time", "year")

# ----------------------------------------------------------------------------
# Setup and data loading
input_properties = {"input_root": configuration.config.get("path", "output")}
tou_columns = configuration.config.get("tariff", "time_of_use_labels")
mm = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0", "new_dimension": "user"}


def create_and_run_user_data_processing_pipeline(user_type, input_filename):
    DataProcessingPipeline("read_and_store", workers=(Read(name=user_type, filename=input_filename, **input_properties),
                                                      TransformCoordinateIntoDimension(name=user_type, **mm),
                                                      # manage hourly data, sum all end users / plants
                                                      Aggregate(name=user_type, aggregate_on={"dim_1": DataKind.MONTH}),
                                                      Apply(name=f"{user_type}_tou_cols",
                                                            operation=lambda x: x.sel({"dim_1": tou_columns})),
                                                      Store(user_type))).execute()


def create_and_run_timeseries_processing_pipeline(profile, input_filename):
    DataProcessingPipeline("read_and_store", workers=(Read(name=profile, filename=input_filename, **input_properties),
                                                      TransformCoordinateIntoDimension(name=f"transform_{profile}",
                                                                                       **mm),
                                                      # Get total production and consumption data
                                                      # Here we manage monthly ToU values, we sum all end users/plants
                                                      Apply(name=profile, operation=lambda x: x.assign_coords(
                                                          dim_1=to_datetime(x.dim_1)).sum(DataKind.USER.value)),
                                                      Store(profile))).execute()

def create_dataset_for_parametric_evaluation():
    """ Get total consumption and production for all users separated months and time of use
    Create a single dataframe for both production and consumptions
    https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
    """
    user_types = ["pv_plants", "families", "users"]
    input_filenames = ["data_plants_tou", "data_families_tou", "data_users_tou"]
    for user_type, filename in zip(user_types, input_filenames):
        create_and_run_user_data_processing_pipeline(user_type, filename)

    profiles = ["pv_profiles", "family_profiles", "user_profiles"]
    input_filenames = ["data_plants_year", "data_families_year", "data_users_year"]
    for profile_type, filename in zip(profiles, input_filenames):
        create_and_run_timeseries_processing_pipeline(profile_type, filename)
    # Create a single dataframe for both production and consumption
    DataProcessingPipeline("concatenate", workers=(
        ArrayConcat(dim=DataKind.USER.value, arrays_to_merge=user_types, coords={DataKind.USER.value: user_types}),
        Rename(dims={"dim_1": DataKind.TOU.value, "group": DataKind.MONTH.value}), Store("tou_months"),
        ArrayConcat(name="merge_profiles", dim=DataKind.USER.value, arrays_to_merge=profiles,
                    coords={DataKind.USER.value: user_types}), Store("energy_year"))).execute()
    ds = DataStore()
    return ds["tou_months"], ds["energy_year"]

# 4.) Merge aggregated consumption/production data into user info dataframe

# ----------------------------------------------------------------------------
# Here we evaluate the number of families to reach the set targets
tou_months, energy_year = create_dataset_for_parametric_evaluation()

# Initialize results
n_fams = []
met_targets = []
scs = []

# Do we need this? We don't know. We can have a PipelineStage for it
# [0, 15, 30, 45, 60]
evaluation = configuration.config.get("parametric_evaluation", "evaluation_parameters")
self_consumption_targets = configuration.config.getarray("parametric_evaluation", "self_consumption_targets")
# Evaluate number of families for each target
sc = 0
for i, sc_target in enumerate(self_consumption_targets):
    # # Skip if previous target was already higher than this
    if i > 0 and sc > sc_target:
        met_targets[-1] = sc_target
        continue

    # # Find number of families to reach target
    n_fam, sc = find_the_optimal_number_of_families_for_sc_ratio(energy_year, sc_target,
                                                                 max(evaluation.number_of_families))

    # Update
    n_fams.append(n_fam)
    met_targets.append(sc_target)
    scs.append(sc)

    # # Exit if targets cannot be reached
    if sc < sc_target:
        print("Exiting loop because requirement cannot be reached.")
        break
    if n_fam >= max(evaluation.number_of_families):
        print("Exiting loop because max families was reached.")
        break

# ----------------------------------------------------------------------------
# Setup to aggregate in time
# Function to evaluate SC with different aggregations in time
time_resolution = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH, sc_week=DataKind.WEEK,
                       sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                       sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])

results = DataFrame(index=n_fams, columns=list(time_resolution.keys()) + ["sc_tou"])
for n_fam in n_fams:
    results.loc[n_fam, 'sc_tou'] = calculate_theoretical_limit_of_self_consumption(tou_months, n_fam)
    for label, tr in time_resolution.items():
        results.loc[n_fam, label] = calculate_sc_for_time_aggregation(energy_year, tr, n_fam)
    calculate_shared_energy(energy_year, n_fam)
    energy_by_day = energy_year.groupby(time_resolution["sc_day"])
    plot_shared_energy(energy_by_day.sum()[DataKind.SHARED],
                       energy_by_day[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(axis="rows"), n_fam)

plot_sci(time_resolution, n_fams, results)

# ----------------------------------------------------------------------------
# %% Here, we check which storage sizes should be considered for each number of families
# Manually insert bess sizes for each number of families
bess_sizes = [[0, ] for _ in n_fams]
print("Manually insert number of sizes for number of families.")
for i, n_fam in enumerate(n_fams):
    if i > 0:
        print("Push 'enter' to copy BESS sizes of the previous n of families")
    bess = input(f"Insert BESS sizes for {n_fam} families:").strip(" ,").split(",")

    if i > 0 and bess == [""]:
        bess_sizes[i] = bess_sizes[i - 1].copy()
        continue
    try:
        bess_sizes[i] += [int(s.strip()) for s in bess]
    except ValueError:
        raise ValueError("Something wrong, retry")

scenarios = DataFrame(data=((np.ones_like(bess_size) * n_fam) for n_fam, bess_size in zip(n_fams, bess_sizes)),
                      columns=[DataKind.NUMBER_OF_FAMILIES, DataKind.BATTERY_SIZE])

scenarios[list(results.keys())] = scenarios[DataKind.NUMBER_OF_FAMILIES].apply(
    lambda x: (r[x] for _, r in results.items()))

Write().write(scenarios, "scenarios")

# Get plants sizes and number of users
n_users = len(ds["data_users"])
data_plants = ds["data_plants"]
pv_sizes = list(data_plants.loc[data_plants[DataKind.USER_TYPE] == 'pv', DataKind.POWER])
if len(pv_sizes) < len(data_plants):
    raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")

# Initialize results
results = DataFrame(index=scenarios.index,
                    columns=['e_prod', 'e_cons', 'e_inj', 'e_with', 'e_sh', 'e_tot', 'em_base', 'sc', 'ss', 'esr',
                             'capex'])

p_prod = energy_year.sel(user=DataKind.PRODUCTION)
p_cons = energy_year.sel(user=DataKind.CONSUMPTION_OF_USERS)
p_fam = energy_year.sel(user=DataKind.CONSUMPTION_OF_FAMILIES)

# Evaluate each scenario
for i, scenario in scenarios.iterrows():
    # Get configuration
    n_fam = scenario[DataKind.NUMBER_OF_FAMILIES]
    bess_size = scenario[DataKind.BATTERY_SIZE]

    # Manage BESS, if present
    p_with = p_cons + n_fam * p_fam
    if bess_size > 0:
        p_bess = manage_bess(p_prod, p_with, bess_size)
        p_inj = p_prod - p_bess
    else:
        p_inj = p_prod.copy()

    # Eval REC
    e_prod = energy(p_prod)
    e_cons = energy(p_with)
    sc, ss, e_sh, e_inj, e_with = eval_physical_parameters(p_inj, p_with)

    # Evaluate emissions
    esr, em_tot, em_base = eval_co2(e_sh, e_cons=e_with, e_inj=e_inj, e_prod=e_prod)

    # Evaluate CAPEX
    capex = eval_capex(pv_sizes, bess_size=bess_size, n_users=n_users + n_fam)

    # Update results
    results.loc[i, :] = (e_prod, e_cons, e_inj, e_with, e_sh, em_tot, em_base, sc, ss, esr, capex)

Write().write(results, "results")
