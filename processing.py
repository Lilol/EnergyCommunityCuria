# Data management
import os

# Visualization
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pandas import DataFrame, concat, read_csv

from utility import configuration
from input.definitions import ColumnName
from input.reader import BillReader
from output.writer import Writer
from utility.day_of_the_week import df_year

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

    # Helper function - fin closest-step integer
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
    # TODO: kill it with fire
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

# Names of the files to load https://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
# TODO: filenames always into config, or kill it with fire
# file_plants = "data_plants.csv"  # list of plants
# file_users = "data_users.csv"  # list of end users
file_plants_tou = "data_plants_tou.csv"  # monthly production data
file_users_tou = "data_users_tou.csv"  # monthly consumption data
file_fam_tou = "data_fam_tou.csv"  # monthly family consumption data
file_plants_year = "data_plants_year.csv"  # one-year hourly production data
file_users_year = "data_users_year.csv"  # one-year hourly consumption data
file_fam_year = "data_fam_year.csv"  # one-year hourly family consumption data

# Load data
data_plants_tou = read_csv(os.path.join(directory_data, file_plants_tou), sep=';')
data_users_tou = read_csv(os.path.join(directory_data, file_users_tou), sep=';')
df_fam_tou = read_csv(os.path.join(directory_data, file_fam_tou), sep=';').drop([ColumnName.USER, ColumnName.YEAR], axis=1).fillna(
    0)
data_plants_year = read_csv(os.path.join(directory_data, file_plants_year), sep=';')
data_users_year = read_csv(os.path.join(directory_data, file_users_year), sep=';')
df_fam = read_csv(os.path.join(directory_data, file_fam_year), sep=';').drop(ColumnName.USER, axis=1)

# ----------------------------------------------------------------------------
# Get total production and consumption data

# Here we manage monthly ToU values, we sum all end users/plants
cols = [ColumnName.MONTH]
df_plants_tou = data_plants_tou.groupby(ColumnName.MONTH).sum()[BillReader._time_of_use_energy_column_names].reset_index()
df_users_tou = data_users_tou.groupby(ColumnName.MONTH).sum()[BillReader._time_of_use_energy_column_names].reset_index()

# We create a single dataframe for both production and consumption
df_months = DataFrame()
for (_, df_prod), (_, df_cons), (_, df_f) in zip(df_plants_tou.iterrows(), df_users_tou.iterrows(),
                                                 df_fam_tou.iterrows()):
    prod = df_prod.loc[BillReader._time_of_use_energy_column_names].values
    cons = df_cons.loc[BillReader._time_of_use_energy_column_names].values
    fam = df_f.loc[BillReader._time_of_use_energy_column_names].values

    df_temp = concat([df_prod[cols]] * len(prod), axis=1).T
    col_tou = 'tou'
    df_temp[col_tou] = arange(len(prod))
    df_temp[ColumnName.PRODUCTION] = prod
    df_temp[ColumnName.CONSUMPTION] = cons
    df_temp[ColumnName.FAMILY] = fam

    df_months = concat((df_months, df_temp), axis=0)

# Here, we manage hourly data, we sum all end users/plants
# cols = [ColumnName.YEAR, ColumnName.SEASON, ColumnName.MONTH, ColumnName.WEEK, ColumnName.DAY_OF_MONTH, ColumnName.DAY_OF_WEEK, ColumnName.DAY_TYPE]
df_plants = df_year.loc[df_year[ColumnName.YEAR] == ref_year, cols]
df_plants = df_plants.merge(data_plants_year.groupby([ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().loc[:, '0':].reset_index(),
                            on=[ColumnName.MONTH, ColumnName.DAY_OF_MONTH])
df_users = df_year.loc[df_year[ColumnName.YEAR] == ref_year, cols]
df_users = df_users.merge(data_users_year.groupby([ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().loc[:, '0':].reset_index(),
                          on=[ColumnName.MONTH, ColumnName.DAY_OF_MONTH])

# We create a single dataframe for both production and consumption
df_hours = DataFrame()
for (_, df_prod), (_, df_cons), (_, df_f) in zip(df_plants.iterrows(), df_users.iterrows(), df_fam.iterrows()):
    prod = df_prod.loc['0':].values
    cons = df_cons.loc['0':].values
    fam = df_f.loc['0':].values

    df_temp = concat([df_prod[cols]] * len(prod), axis=1).T
    df_temp[ColumnName.HOUR] = np.arange(len(prod))
    df_temp[ColumnName.PRODUCTION] = prod
    df_temp[ColumnName.CONSUMPTION] = cons
    df_temp[ColumnName.FAMILY] = fam

    df_hours = concat((df_hours, df_temp), axis=0)

# ----------------------------------------------------------------------------
# Here we evaluate the number of families to reach the set targets

# Get data arrays
p_prod = df_hours[ColumnName.PRODUCTION].values
p_cons = df_hours[ColumnName.CONSUMPTION].values
p_fam = df_hours[ColumnName.FAMILY].values

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
def sc_lim_tou(n_fam):
    prod = df_months[ColumnName.PRODUCTION]
    cons = df_months[ColumnName.CONSUMPTION]
    fam = df_months[ColumnName.FAMILY] * n_fam
    e_shared = np.minimum(prod, cons + fam).sum()
    e_prod = prod.sum()
    return e_shared / e_prod


# Function to evaluate SC with different aggregations in time
def aggregated_sc(groupby, n_fam):
    """Evaluate SC with given temporal aggregation and number of families."""
    # Get values
    cols = [ColumnName.PRODUCTION, ColumnName.CONSUMPTION, ColumnName.FAMILY]
    prod, cons, fam = df_hours.groupby(groupby).sum()[cols].values.T
    cons = cons + fam * n_fam

    # Evaluate shared energy
    shared = np.minimum(prod, cons)

    # Return SC
    return np.sum(shared) / np.sum(prod)


# Setup to aggregate in time
groupbys = dict(sc_year=ColumnName.YEAR, sc_season=ColumnName.SEASON, sc_month=ColumnName.MONTH, sc_week=ColumnName.WEEK,
                sc_day=[ColumnName.MONTH, ColumnName.DAY_OF_MONTH], sc_hour=[ColumnName.MONTH, ColumnName.DAY_OF_MONTH,
                                                                             ColumnName.HOUR])

results = {label: [] for label in groupbys.keys()}
results['sc_tou'] = []

# for n_fam, sc in zip(n_fams, scs):
for n_fam in n_fams:
    results['sc_tou'].append(sc_lim_tou(n_fam))
    for label, groupby in groupbys.items():
        sc = aggregated_sc(groupby, n_fam)
        results[label].append(sc)

plt.figure()
for label in groupbys:
    plt.plot(n_fams, results[label], label=label)
plt.plot(n_fams, results['sc_tou'], label='sc_tou', color='lightgrey', ls='--')
# plt.scatter(n_fams, scs, label='evaluated')
plt.xlabel('Numero famiglie')
plt.ylabel('SCI')
plt.legend()
plt.show()
plt.close()

# ----------------------------------------------------------------------------
# %% Here, we check which storage sizes should be considered for each number of
# families

for n_fam in n_fams:
    df_hhours = df_hours.copy()

    prod, cons, fam = df_hhours[[ColumnName.PRODUCTION, ColumnName.CONSUMPTION, ColumnName.FAMILY]].values.T
    inj = prod
    with_ = cons + fam * n_fam
    shared = np.minimum(inj, with_)

    df_hhours['injections'] = inj
    df_hhours['withdrawals'] = with_
    df_hhours['shared'] = shared

    sh1 = df_hhours.groupby([ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum()['shared'].values
    sh2 = np.minimum(*df_hhours.groupby([ColumnName.MONTH, ColumnName.DAY_OF_MONTH])[['injections', 'withdrawals']].sum().values.T)

    plt.figure()
    plt.plot(np.diff(sorted(sh2 - sh1)), label=str(n_fam), color='lightgrey')
    plt.yticks([])
    plt.xlabel('Numero giorni dell\'anno')

    plt.twinx().plot(sorted(sh2 - sh1), label=str(n_fam))
    plt.ylabel('Gap tra energia condivisa oraria e giornaliera (kWh)')
    plt.gca().yaxis.set_label_position("left")
    plt.gca().yaxis.tick_left()

    plt.title(f"Numero famiglie: {int(n_fam)}")
    plt.show()
    plt.close()

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


# TODO: use lambda or kill it with fire
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


Writer().write(scenarios, "scenarios")
