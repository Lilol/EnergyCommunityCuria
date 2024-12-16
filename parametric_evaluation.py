import numpy as np
from pandas import DataFrame, to_datetime

from analysis.evaluating import eval_co2, eval_capex, energy, eval_rec, manage_bess, \
    find_the_optimal_number_of_families_for_sc_ratio, calculate_shared_energy, \
    calculate_theoretical_limit_of_self_consumption, calculate_sc_for_specific_time_aggregation
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
# Directory with files
# 1.) Reading files

# Names of the files to load
# https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
input_properties = {"input_root": configuration.config.get("path", "output")}
tou_columns = configuration.config.get("tariff", "time_of_use_labels")
mm = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0", "new_dimension": "user"}
DataProcessingPipeline("read_and_store", workers=(
    Read(name="pv_plants", filename="data_plants_tou", **input_properties),
    TransformCoordinateIntoDimension(name="pv_plants", **mm),
    Aggregate(name="pv_plants", aggregate_on={"dim_1": DataKind.MONTH}),
    Apply(name="pv_select_tou_cols", operation=lambda x: x.sel({"dim_1": tou_columns})),
    Store("pv_plants"),
    Read(name="users", filename="data_users_tou", **input_properties),
    TransformCoordinateIntoDimension(name="users", **mm),
    Aggregate(name="users", aggregate_on={"dim_1": DataKind.MONTH}),
    Apply(name="users_select_tou_cols", operation=lambda x: x.sel({"dim_1": tou_columns})),
    Store("users"),
    Read(name="families", filename="data_families_tou", **input_properties),
    TransformCoordinateIntoDimension(name="families", **mm),
    Aggregate(name="families", aggregate_on={"dim_1": DataKind.MONTH}),
    Apply(name="families_select_tou_cols", operation=lambda x: x.sel({"dim_1": tou_columns})),
    Store("families"),
    ArrayConcat(name="merge_consumption_production", dim=DataKind.USER.value, arrays_to_merge=["pv_plants", "families", "users"],
                coords={DataKind.USER.value: ["plants", "families", "users"]}),
    Rename(dims={"dim_1": DataKind.TOU.value, "group": DataKind.MONTH.value}),
    Store("tou_months"),
    Read(name="pv_profiles", filename="data_plants_year", **input_properties),
    TransformCoordinateIntoDimension(name="transform_pv_profiles", **mm),
    Apply(name="pv_profiles", operation=lambda x: x.assign_coords(dim_1=to_datetime(x.dim_1)).sum(DataKind.USER.value)),
    Store("pv_profiles"),
    Read(name="user_profiles", filename="data_users_year", **input_properties),
    TransformCoordinateIntoDimension(name="transform_user_profiles", **mm),
    Apply(name="user_profiles",operation=lambda x: x.assign_coords(dim_1=to_datetime(x.dim_1)).sum(DataKind.USER.value)),
    Store("user_profiles"),
    Read(name="family_profiles", filename="data_families_year", **input_properties),
    TransformCoordinateIntoDimension(name="transform_family_profiles", **mm),
    Apply(name="family_profiles", operation=lambda x: x.assign_coords(dim_1=to_datetime(x.dim_1)).sum(DataKind.USER.value)),
    Store("family_profiles"),
    ArrayConcat(name="merge_profiles", dim=DataKind.USER.value,
                arrays_to_merge=["pv_profiles", "family_profiles", "user_profiles"],
                coords={DataKind.USER.value: ["plants", "families", "users"]}),
    Store("energy_year"))).execute()

# ----------------------------------------------------------------------------
# Get total production and consumption data
# Here we manage monthly ToU values, we sum all end users/plants
# 2.) Get total consumption and production for all users separated months and time of use
# We create a single dataframe for both production and consumption
# 3.) 2D frame with rows: TOU time slots, cols are families, users and PV producers

ds = DataStore()
tou_months = ds["tou_months"]
energy_year = ds["energy_year"]

# 4.) Merge aggregated consumption/production data into user info dataframe
# Here, we manage hourly data, we sum all end users/plants
# We create a single dataframe for both production and consumption

# ----------------------------------------------------------------------------
# Here we evaluate the number of families to reach the set targets

# Initialize results
n_fams = []
met_targets = []
scs = []

# Do we need this? We don't know. We can have a PipelineStep for it
# [0, 15, 30, 45, 60]
evaluation = configuration.config.get("parametric_evaluation", "to_evaluate")
n_fams_ = []
sc_targets = []
# Evaluate number of families for each target
sc = 0
for i, sc_target in enumerate(np.arange(0, 1, 0.1)):
    # # Skip if previous target was already higher than this
    if i > 0 and sc > sc_target:
        met_targets[-1] = sc_target
        continue

    # # Find number of families to reach target
    n_fam, sc = find_the_optimal_number_of_families_for_sc_ratio(energy_year, sc_target, max(n_fams_))

    # Update
    n_fams.append(n_fam)
    met_targets.append(sc_target)
    scs.append(sc)

    # # Exit if targets cannot be reached
    if sc < sc_target:
        print("Exiting loop because requirement cannot be reached.")
        break
    if n_fam == max(n_fams_):
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
        results.loc[n_fam, label] = calculate_sc_for_specific_time_aggregation(energy_year, tr, n_fam)

plot_sci(time_resolution, n_fams, results)

for n_fam in n_fams:
    calculate_shared_energy(energy_year, n_fam)
    sh1 = energy_year.groupby(time_resolution["sc_day"]).sum()[DataKind.SHARED]
    sh2 = energy_year.groupby(time_resolution["sc_day"])[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].sum().min(
        axis="rows")
    plot_shared_energy(sh1, sh2, n_fam)

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

p_prod = energy_year[DataKind.PRODUCTION]
p_cons = energy_year[DataKind.CONSUMPTION_OF_RESIDENTIAL]
p_fam = energy_year[DataKind.CONSUMPTION_OF_FAMILIES]

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
    sc, ss, e_sh = eval_rec(p_inj, p_with)
    e_inj = e_sh / sc
    e_with = e_sh / ss

    # Evaluate emissions
    esr, em_tot, em_base = eval_co2(e_sh, e_cons=e_with, e_inj=e_inj, e_prod=e_prod)

    # Evaluate CAPEX
    capex = eval_capex(pv_sizes, bess_size=bess_size, n_users=n_users + n_fam)

    # Update results
    results.loc[i, :] = (e_prod, e_cons, e_inj, e_with, e_sh, em_tot, em_base, sc, ss, esr, capex)

Write().write(results, "results")
