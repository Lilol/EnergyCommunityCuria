import numpy as np
import xarray as xr
from pandas import DataFrame, to_datetime

from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from input.definitions import DataKind
from input.read import Read
from output.write import Write
from transform.transform import TransformCoordinateIntoDimension
from utility import configuration
from visualization.processing_visualization import plot_sci, plot_shared_energy

ref_year = configuration.config.getint("time", "year")


def eval_sc(df, n_fam):
    calculate_shared_energy(df, n_fam)
    return calculate_sc(df)


# ----------------------------------------------------------------------------
# Utility functions
# Helper function - find closest-step integer
def find_closer(n_fam, step):
    """Return closer integer to n_fam, considering the step."""
    if n_fam % step == 0:
        return n_fam
    if n_fam % step >= step / 2:
        return (n_fam // step) + 1
    else:
        return n_fam // step


def find_best_value(df, n_fam_high, n_fam_low, step, sc):
    # Stopping criterion (considering that n_fam is integer)
    if n_fam_high - n_fam_low <= step:
        print("Procedure ended without exact match.")
        return n_fam_high, eval_sc(df, n_fam_high)

    # Bisection of the current space
    n_fam_mid = find_closer((n_fam_low + n_fam_high) / 2, step)
    sc_mid = eval_sc(df, n_fam_mid)

    # Evaluate and update
    if sc_mid - sc == 0:  # Check if exact match is found
        print("Found exact match.")
        return n_fam_mid, sc_mid

    elif sc_mid < sc:
        return find_best_value(df, n_fam_high, n_fam_mid, step, sc)
    else:
        return find_best_value(df, n_fam_mid, n_fam_low, step, sc)


# Find the optimal number of families to satisfy a given self-consumption ratio
def find_the_optimal_number_of_families_for_sc_ratio(df, sc, n_fam_max, step=25):
    """
    Finds the optimal number of families to satisfy a given self-consumption
    ratio.

    Parameters:
    - sc (float): Target self consumption ratio.
    - n_fam_max (int): Maximum number of families.
    - p_plants (numpy.ndarray): Array of power values from plants.
    - p_users (numpy.ndarray): Array of power values consumed by users.
    - p_fam (numpy.ndarray): Array of power values consumed by each family.
    - step (int): Step in the number of families.

    Returns:
    - tuple: Tuple containing the optimal number of families and the
        corresponding shared energy ratio.
    """

    # Evaluate starting point
    n_fam_low = 0
    sc_low = eval_sc(df, n_fam_low)
    if sc_low >= sc:  # Check if requirement is already satisfied
        print("Requirement already satisfied!")
        return n_fam_low, sc_low

    # Evaluate point that can be reached
    n_fam_high = n_fam_max
    sc_high = eval_sc(df, n_fam_high)
    if sc_high <= sc:  # Check if requirement is satisfied
        print("Requirement cannot be satisfied!")
        return n_fam_high, sc_high

    # Loop to find best value
    return find_best_value(df, n_fam_high, n_fam_low, step, sc)


# ----------------------------------------------------------------------------
# Setup and data loading
# Directory with files
# 1.) Reading files

# Names of the files to load
# https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
input_properties = {"input_root": configuration.config.get("path", "output")}

mm = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0", "new_dimension": "user"}
DataProcessingPipeline("read_and_store", workers=(
    Read(name="pv_plants", filename="data_plants_tou", **input_properties),
    TransformCoordinateIntoDimension("pv_plants", **mm), Store("pv_plants"),
    Read(name="users", filename="data_users_tou", **input_properties), TransformCoordinateIntoDimension("users", **mm),
    Store("users"), Read(name="families", filename="data_families_tou", **input_properties),
    TransformCoordinateIntoDimension("families", **mm), Store("families"),
    Read(name="pv_profiles", filename="data_plants_year", **input_properties),
    TransformCoordinateIntoDimension("pv_profiles", **mm), Store("pv_profiles"),
    Read(name="user_profiles", filename="data_users_year", **input_properties),
    TransformCoordinateIntoDimension("user_profiles", **mm), Store("user_profiles"),
    Read(name="family_profiles", filename="data_families_year", **input_properties),
    TransformCoordinateIntoDimension("family_profiles", **mm), Store("family_profiles"))).execute()

# ----------------------------------------------------------------------------
# Get total production and consumption data
# Here we manage monthly ToU values, we sum all end users/plants
# 2.) Get total consumption and production for all users separated months and time of use
ds = DataStore()
tou_columns = configuration.config.get("tariff", "time_of_use_labels")
plants = ds["pv_plants"]
plants = plants.groupby(plants.sel({"dim_1": DataKind.MONTH})).sum().sel({"dim_1": tou_columns})
users = ds["users"]
users = users.groupby(users.sel({"dim_1": DataKind.MONTH})).sum().sel({"dim_1": tou_columns})
families = ds["families"]
families = families.groupby(families.sel({"dim_1": DataKind.MONTH})).sum().sel({"dim_1": tou_columns})

# We create a single dataframe for both production and consumption
# 3.) 2D frame with rows: TOU time slots, cols are families, users and PV producers

tou_months = xr.concat([users, families, plants], dim="user").assign_coords(
    {"user": ["users", "families", "plants"]}).rename({"dim_1": "tou", "group": "month"})

# 4.) Merge aggregated consumption/production data into user info dataframe
# Here, we manage hourly data, we sum all end users/plants

# 5.) I thought we already did this, but this is for hourly data, not aggregated
# We create a single dataframe for both production and consumption
plants_year = ds["pv_profiles"]
plants_year = plants_year.assign_coords(dim_1=to_datetime(plants_year.dim_1)).sum("user")

users_year = ds["user_profiles"]
users_year = users_year.assign_coords(dim_1=to_datetime(users_year.dim_1)).sum("user")

families_year = ds["family_profiles"]
families_year = families_year.assign_coords(dim_1=to_datetime(families_year.dim_1)).sum("user")

energy_year = xr.concat([users_year, families_year, plants_year], dim="user").assign_coords(
    {"user": ["users", "families", "plants"]})

# ----------------------------------------------------------------------------
# Here we evaluate the number of families to reach the set targets

# Get data arrays
p_prod = energy_year[DataKind.PRODUCTION].values
p_cons = energy_year[DataKind.CONSUMPTION_OF_RESIDENTIAL].values
p_fam = energy_year[DataKind.CONSUMPTION_OF_FAMILIES].values

# Initialize results
n_fams = []
met_targets = []
scs = []

# Do we need this? We don't know. We can have a PipelineStep for it
n_fams_ = configuration.config.getarray("rec", "number_of_families", int)  # [0, 15, 30, 45, 60]
sc_targets = [0, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# Evaluate number of families for each target
sc = 0
for i, sc_target in enumerate(sc_targets):
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
        print("Exiting loop because max families reached.")
        break


# ----------------------------------------------------------------------------
# %% Here, we evaluate the need for daily/seasonal storage depending on the
# number of families
def calc_sum_consumption(df, n_fam):
    df[DataKind.CONSUMPTION] = df[DataKind.CONSUMPTION_OF_FAMILIES] * n_fam + df[DataKind.CONSUMPTION_OF_RESIDENTIAL]


def calculate_shared_energy(df, n_fam):
    calc_sum_consumption(df, n_fam)
    df[DataKind.SHARED] = df[[DataKind.PRODUCTION, DataKind.CONSUMPTION]].min(axis="rows")


def calculate_sc(df):
    return df[DataKind.SHARED].sum() / df[DataKind.PRODUCTION].sum()


# Function to evaluate a "theoretical" limit to shared energy/self-consumption
# given the ToU monthly energy values
def calculate_theoretical_limit_of_self_consumption(df_months, n_fam):
    calculate_shared_energy(df_months, n_fam)
    return calculate_sc(df_months)


# Function to evaluate SC with different aggregations in time
def calculate_sc_for_specific_time_aggregation(df_hours, time_resolution, n_fam):
    """Evaluate SC with given temporal aggregation and number of families."""
    calculate_shared_energy(df_hours.groupby(time_resolution).sum(), n_fam)
    return calculate_sc(df_hours)


# Setup to aggregate in time
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
    sh2 = energy_year.groupby(time_resolution["sc_day"]).sum()[[DataKind.CONSUMPTION, DataKind.PRODUCTION]].min(
        axis="rows")
    plot_shared_energy(sh1, sh2, n_fam)
# ----------------------------------------------------------------------------
# %% Here, we check which storage sizes should be considered for each number of families
# Manually insert bess sizes for each number of families
bess_sizes = [[0, ] for _ in n_fams]
for i, n_fam in enumerate(n_fams):
    print("Manually insert number of sizes for this number of families.")
    if i > 0:
        print("Push 'enter' to copy BESS sizes of the previous n of families")
    bess = input(f"Insert BESS sizes for {n_fam} families:")
    bess = bess.strip(" ,").split(",")

    if i > 0 and bess == [""]:
        bess_sizes[i] = bess_sizes[i - 1].copy()
        continue
    try:
        bess = [int(s.strip()) for s in bess]
    except ValueError:
        raise ValueError("Something wrong, retry")

    bess_sizes[i] += bess

scenarios = DataFrame(
    data=((np.ones_like(bess_size) * n_fam), bess_size for n_fam, bess_size in zip(n_fams, bess_sizes)),
    columns=[DataKind.NUMBER_OF_FAMILIES, DataKind.BATTERY_SIZE])

scenarios[list(results.keys())] = scenarios[DataKind.NUMBER_OF_FAMILIES].apply(
    lambda x: (r for i, r in results.items()))

Write().write(scenarios, "scenarios")
