# ----------------------------------------------------------------------------
# Import statement

# Data management
import os

# Visualization
import matplotlib.pyplot as plt

# TODO: absolutely NOT
from common import *  # common variables


# Data processing


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
# TODO: to config
# n_fam_max = 200  # (SETUP)
# sc_targets = [0, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # 0 needed
n_fams_ = [0, 25, 50, 75, 100]  # [0, 15, 30, 45, 60]

# Directory with files
directory_data = 'DatiProcessati'

# Names of the files to loadhttps://www.arera.it/dati-e-statistiche/dettaglio/analisi-dei-consumi-dei-clienti-domestici
# file_plants = "data_plants.csv"  # list of plants
# file_users = "data_users.csv"  # list of end users
# TODO: filenames always into config
file_plants_tou = "data_plants_tou.csv"  # monthly production data
file_users_tou = "data_users_tou.csv"  # monthly consumption data
file_fam_tou = "data_fam_tou.csv"  # monthly family consumption data
file_plants_year = "data_plants_year.csv"  # one-year hourly production data
file_users_year = "data_users_year.csv"  # one-year hourly consumption data
file_fam_year = "data_fam_year.csv"  # one-year hourly family consumption data

# Load data
data_plants_tou = pd.read_csv(os.path.join(directory_data, file_plants_tou), sep=';')
data_users_tou = pd.read_csv(os.path.join(directory_data, file_users_tou), sep=';')
df_fam_tou = pd.read_csv(os.path.join(directory_data, file_fam_tou), sep=';').drop([col_user, col_year], axis=1).fillna(
    0)
data_plants_year = pd.read_csv(os.path.join(directory_data, file_plants_year), sep=';')
data_users_year = pd.read_csv(os.path.join(directory_data, file_users_year), sep=';')
df_fam = pd.read_csv(os.path.join(directory_data, file_fam_year), sep=';').drop(col_user, axis=1)

# ----------------------------------------------------------------------------
# Get total production and consumption data

# Here we manage monthly ToU values, we sum all end users/plants
cols = [col_month]
df_plants_tou = data_plants_tou.groupby(col_month).sum()[cols_tou_energy].reset_index()
df_users_tou = data_users_tou.groupby(col_month).sum()[cols_tou_energy].reset_index()

# We create a single dataframe for both production and consumption
df_months = pd.DataFrame()
for (_, df_prod), (_, df_cons), (_, df_f) in zip(df_plants_tou.iterrows(), df_users_tou.iterrows(),
                                                 df_fam_tou.iterrows()):
    prod = df_prod.loc[cols_tou_energy].values
    cons = df_cons.loc[cols_tou_energy].values
    fam = df_f.loc[cols_tou_energy].values

    df_temp = pd.concat([df_prod[cols]] * len(prod), axis=1).T
    col_tou = 'tou'
    df_temp[col_tou] = np.arange(len(prod))
    df_temp['production'] = prod
    df_temp['consumption'] = cons
    df_temp['family'] = fam

    df_months = pd.concat((df_months, df_temp), axis=0)

# Here, we manage hourly data, we sum all end users/plants
cols = [col_year, col_season, col_month, col_week, col_day, col_dayweek, col_daytype]
df_plants = df_year.loc[df_year[col_year] == ref_year, cols]
df_plants = df_plants.merge(data_plants_year.groupby([col_month, col_day]).sum().loc[:, '0':].reset_index(),
                            on=[col_month, col_day])
df_users = df_year.loc[df_year[col_year] == ref_year, cols]
df_users = df_users.merge(data_users_year.groupby([col_month, col_day]).sum().loc[:, '0':].reset_index(),
                          on=[col_month, col_day])

# We create a single dataframe for both production and consumption
df_hours = pd.DataFrame()
for (_, df_prod), (_, df_cons), (_, df_f) in zip(df_plants.iterrows(), df_users.iterrows(), df_fam.iterrows()):
    prod = df_prod.loc['0':].values
    cons = df_cons.loc['0':].values
    fam = df_f.loc['0':].values

    df_temp = pd.concat([df_prod[cols]] * len(prod), axis=1).T
    col_hour = 'hour'
    df_temp[col_hour] = np.arange(len(prod))
    df_temp['production'] = prod
    df_temp['consumption'] = cons
    df_temp['family'] = fam

    df_hours = pd.concat((df_hours, df_temp), axis=0)

# ----------------------------------------------------------------------------
# Here we evaluate the number of families to reach the set targets

# Get data arrays
p_prod = df_hours['production'].values
p_cons = df_hours['consumption'].values
p_fam = df_hours['family'].values

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
    prod = df_months['production']
    cons = df_months['consumption']
    fam = df_months['family'] * n_fam
    e_shared = np.minimum(prod, cons + fam).sum()
    e_prod = prod.sum()
    return e_shared / e_prod


# Function to evaluate SC with different aggregation in time
def aggregated_sc(groupby, n_fam):
    """Evaluate SC with given temporal aggregation and number of families."""
    # Get values
    cols = ['production', 'consumption', 'family']
    prod, cons, fam = df_hours.groupby(groupby).sum()[cols].values.T
    cons = cons + fam * n_fam

    # Evaluate shared energy
    shared = np.minimum(prod, cons)

    # Return SC
    return np.sum(shared) / np.sum(prod)


# Setup to aggregate in time
groupbys = dict(sc_year=col_year, sc_season=col_season, sc_month=col_month, sc_week=col_week,
                sc_day=[col_month, col_day], sc_hour=[col_month, col_day, col_hour])

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

# ----------------------------------------------------------------------------
# %% Here, we check which storage sizes should be considered for each number of
# families

for n_fam in n_fams:
    df_hhours = df_hours.copy()

    prod, cons, fam = df_hhours[['production', 'consumption', 'family']].values.T
    inj = prod
    with_ = cons + fam * n_fam
    shared = np.minimum(inj, with_)

    df_hhours['injections'] = inj
    df_hhours['withdrawals'] = with_
    df_hhours['shared'] = shared

    sh1 = df_hhours.groupby([col_month, col_day]).sum()['shared'].values
    sh2 = np.minimum(*df_hhours.groupby([col_month, col_day])[['injections', 'withdrawals']].sum().values.T)

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

scenarios = pd.DataFrame(scenarios)


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

scenarios.to_csv(os.path.join(directory_data, "scenarios.csv"), sep=';', index=False)
