# Data management

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import DataFrame, concat, to_datetime

from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from input.definitions import DataKind
from input.read import Read
from output.write import Write
from transform.transform import TransformCoordinateIntoDimension
from utility import configuration
from utility.day_of_the_week import df_year
from visualization.processing_visualization import plot_sci, plot_shared_energy

# Data processing
ref_year = configuration.config.getint("time", "year")


# ----------------------------------------------------------------------------
# Useful functions


# Find the optimal number of families to satisfy a given self-consumption ratio
def find_n_fam(sc, n_fam_max, p_plants, p_users, p_fam, step=25):
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

    # Helper function - find closest-step integer
    def find_closer(n_fam):
        """Return closer integer to n_fam, considering the step."""
        if n_fam % step == 0:
            return n_fam
        if n_fam % step >= step / 2:
            return (n_fam // step) + 1
        else:
            return n_fam // step

    # Helper function - evaluate SC ratio
    def eval_sc(n_fam):
        """
        Helper function to evaluate the shared energy ratio for a given number
        of families.

        Parameters:
        - n_fam (int): Number of families.

        Returns:
        - float: Shared energy ratio.
        """
        p_inj = p_plants.copy()
        p_with = p_users + n_fam * p_fam
        sc = np.minimum(p_inj, p_with).sum() / np.sum(p_inj)
        return sc

    # Evaluate starting point
    n_fam_low = 0
    sc_low = eval_sc(n_fam_low)
    if sc_low - sc >= 0:  # Check if requirement is already satisfied
        print("Requirement already satisfied!")
        return n_fam_low, sc_low

    # Evaluate point that can be reached
    n_fam_high = n_fam_max
    sc_high = eval_sc(n_fam_high)
    if sc_high - sc <= 0:  # Check if requirement is satisfied
        print("Requirement cannot be satisfied!")
        return n_fam_high, sc_high

    # Loop to find best value
    while True:
        # Stopping criterion (considering that n_fam is integer)
        if n_fam_high - n_fam_low <= step:
            print("Procedure ended without exact match.")
            return n_fam_high, sc_high

        # Bisection of the current space
        n_fam_mid = find_closer((n_fam_low + n_fam_high) / 2)
        sc_mid = eval_sc(n_fam_mid)

        # Evaluate and update
        if sc_mid - sc == 0:  # Check if exact match is found
            print("Found exact match.")
            return n_fam_mid, sc_mid
        elif sc_mid - sc < 0:
            n_fam_low, sc_low = n_fam_mid, sc_mid  # Update lower bound
        else:
            n_fam_high, sc_high = n_fam_mid, sc_mid  # Update upper bound


# ----------------------------------------------------------------------------
# Setup and data loading
# n_fam_max = 200  # (SETUP)
# sc_targets = [0, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # 0 needed
n_fams_ = configuration.config.getarray("rec", "number_of_families", int)  # [0, 15, 30, 45, 60]

# Directory with files
directory_data = 'DatiProcessati'

# 1.) Reading previously dumped files

# Names of the files to load https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
# file_plants = "data_plants.csv"  # list of plants
# file_users = "data_users.csv"  # list of end users

input_properties = {"input_root": configuration.config.get("path", "output")}

mm = {"coordinate": {"dim_1": DataKind.USER}, "to_replace_dimension": "dim_0", "new_dimension": "user"}
DataProcessingPipeline("read_and_store", workers=(
    Read(name="pv_plants", filename="data_plants_tou", **input_properties), TransformCoordinateIntoDimension("pv_plants", **mm),
    Store("pv_plants"), Read(name="users", filename="data_users_tou", **input_properties),
    TransformCoordinateIntoDimension("users", **mm), Store("users"),
    Read(name="families", filename="data_families_tou", **input_properties), TransformCoordinateIntoDimension("families", **mm),
    Store("families"), Read(name="pv_profiles", filename="data_plants_year", **input_properties), TransformCoordinateIntoDimension("pv_profiles", **mm), Store("pv_profiles"),
    Read(name="user_profiles", filename="data_users_year", **input_properties), TransformCoordinateIntoDimension("user_profiles", **mm), Store("user_profiles"),
    Read(name="family_profiles", filename="data_families_year", **input_properties), TransformCoordinateIntoDimension("family_profiles", **mm),
    Store("family_profiles"))).execute()

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
# cols = [ColumnName.YEAR, ColumnName.SEASON, ColumnName.MONTH, ColumnName.WEEK, ColumnName.DAY_OF_MONTH, ColumnName.DAY_OF_WEEK, ColumnName.DAY_TYPE]

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
p_cons = energy_year[DataKind.CONSUMPTION].values
p_fam = energy_year[DataKind.FAMILY].values

# Initialize results
n_fams = []
met_targets = []
# scs = []

# Evaluate number of families for each target
# for i, sc_target in enumerate(sc_targets):
for n_fam in n_fams_:
    # # Skip if previous target was already higher than this
    # if i > 0 and sc > sc_target:
    #     met_targets[-1] = sc_target
    #     continue

    # # Find number of families to reach target
    # n_fam, sc = find_n_fam(sc_target, n_fam_max, p_prod, p_cons, p_fam)

    # Update
    n_fams.append(n_fam)  # met_targets.append(sc_target)  # scs.append(sc)

    # # Exit if targets cannot be reached  # if sc < sc_target:  #     print("Exiting loop because requirement cannot be reached.")  #     break  # if n_fam == n_fam_max:  #     print("Exiting loop because max families reached.")  #     break


# ----------------------------------------------------------------------------
# %% Here, we evaluate the need for daily/seasonal storage depending on the
# number of families

# Function to evaluate a "theoretical" limit to shared energy/self-consumption
# given the ToU monthly energy values
def calculate_theoretical_limit_of_self_consumption(df_months, n_fam):
    prod = df_months[DataKind.PRODUCTION]
    cons = df_months[DataKind.CONSUMPTION]
    fam = df_months[DataKind.FAMILY] * n_fam
    e_shared = np.minimum(prod, cons + fam).sum()
    e_prod = prod.sum()
    return e_shared / e_prod


# Function to evaluate SC with different aggregations in time
def aggregate_sc(df_hours, groupby, n_fam):
    """Evaluate SC with given temporal aggregation and number of families."""
    # Get values
    cols = [DataKind.PRODUCTION, DataKind.CONSUMPTION, DataKind.FAMILY]
    prod, cons, fam = df_hours.groupby(groupby).sum()[cols].values.T
    cons = cons + fam * n_fam

    # Evaluate shared energy
    shared = np.minimum(prod, cons)

    # Return SC
    return np.sum(shared) / np.sum(prod)


# Setup to aggregate in time
groupbys = dict(sc_year=DataKind.YEAR, sc_season=DataKind.SEASON, sc_month=DataKind.MONTH, sc_week=DataKind.WEEK,
                sc_day=[DataKind.MONTH, DataKind.DAY_OF_MONTH],
                sc_hour=[DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR])

results = {label: [] for label in groupbys.keys()}
results['sc_tou'] = []

# for n_fam, sc in zip(n_fams, scs):
for n_fam in n_fams:
    results['sc_tou'].append(calculate_theoretical_limit_of_self_consumption(n_fam))
    for label, groupby in groupbys.items():
        sc = aggregate_sc(groupby, n_fam)
        results[label].append(sc)

plot_sci(groupbys, n_fams, results, label)

# ----------------------------------------------------------------------------
# %% Here, we check which storage sizes should be considered for each number of
# families

for n_fam in n_fams:
    df_hhours = energy_year.copy()

    prod, cons, fam = df_hhours[[DataKind.PRODUCTION, DataKind.CONSUMPTION, DataKind.FAMILY]].values.T
    inj = prod
    with_ = cons + fam * n_fam
    shared = np.minimum(inj, with_)

    df_hhours['injections'] = inj
    df_hhours['withdrawals'] = with_
    df_hhours['shared'] = shared

    sh1 = df_hhours.groupby([DataKind.MONTH, DataKind.DAY_OF_MONTH]).sum()['shared'].values
    sh2 = np.minimum(
        *df_hhours.groupby([DataKind.MONTH, DataKind.DAY_OF_MONTH])[['injections', 'withdrawals']].sum().values.T)

    plot_shared_energy(sh1, sh2, n_fam)

# %% Manually insert bess sizes for each number of families
bess_sizes = [[0] for _ in n_fams]
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
    except ValueError("Something wrong, retry"):
        raise ValueError

    bess_sizes[i] += bess

bess_sizes = [sorted(set(b)) for b in bess_sizes]

scenarios = dict(n_fam=[], bess_size=[])
for i, n_fam in enumerate(n_fams):
    scenarios['n_fam'] += [int(n_fam) for _ in bess_sizes[i]]
    scenarios['bess_size'] += bess_sizes[i]

scenarios = DataFrame(scenarios)

# Add limit SC
# sc_target = {n_fam: met_targets[i] for i, n_fam in enumerate(n_fams)}
sc_tou = {n_fam: results['sc_tou'][i] for i, n_fam in enumerate(n_fams)}
sc_day = {n_fam: results['sc_day'][i] for i, n_fam in enumerate(n_fams)}
sc_week = {n_fam: results['sc_week'][i] for i, n_fam in enumerate(n_fams)}
sc_month = {n_fam: results['sc_month'][i] for i, n_fam in enumerate(n_fams)}
sc_season = {n_fam: results['sc_season'][i] for i, n_fam in enumerate(n_fams)}
sc_year = {n_fam: results['sc_year'][i] for i, n_fam in enumerate(n_fams)}

# scenarios['sc_target'] = scenarios['n_fam'].map(sc_target)
scenarios['sc_tou'] = scenarios['n_fam'].map(sc_tou)
scenarios['sc_day'] = scenarios['n_fam'].map(sc_day)
scenarios['sc_week'] = scenarios['n_fam'].map(sc_week)
scenarios['sc_month'] = scenarios['n_fam'].map(sc_month)
scenarios['sc_season'] = scenarios['n_fam'].map(sc_season)
scenarios['sc_year'] = scenarios['n_fam'].map(sc_year)

Write().write(scenarios, "scenarios")
