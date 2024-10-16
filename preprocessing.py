# ----------------------------------------------------------------------------
# Import statement

# Data management
import os

import numpy as np
# Plotting
from matplotlib import pyplot as plt
from pandas import DataFrame, concat, read_csv, date_range, timedelta_range

#
import common as cm
from approach_gse import evaluate as eval_profiles_gse
#
from utils import eval_x, eval_y_from_year


# ----------------------------------------------------------------------------
# Useful functions

# TODO: option for final time resolution!!!
# TODO: option for data processing methodology (will be more in the future)

# Reshape array of one-year data by days
def reshape_array_by_year(array, year):
    """Reshapes an array to (number of days in the year, k), accounting for
    leap years."""

    # Check if the given year is a leap year
    is_leap_year = ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)

    # Get the number of days in the year based on whether it's a leap year
    num_days_in_year = 366 if is_leap_year else 365

    # Calculate k based on the size of the array
    k = len(array) // num_days_in_year

    # Assert that k is an integer
    assert isinstance(k, int), "The calculated value of k must be an integer."

    # Reshape the array based on the number of days in the year and k
    reshaped_array = np.reshape(array, (num_days_in_year, k))

    return reshaped_array


# ----------------------------------------------------------------------------
# Setup

# TODO: config file
# Define the main directory that contains all the municipality folders
directory_data = "DatiComuni"

# Define the name of the directory for the output files
directory_out = "DatiProcessati"

# Define the names of the files to load for each municipality folder
file_plants = "lista_impianti.csv"  # file with list of plants
directory_production = "PVSOL"  # "PVGIS"  #  # folder with the hourly production data
file_users = "lista_pod.csv"  # file with list of end-users
file_users_bills = "dati_bollette.csv"  # file with monthly consumption data

# Perform graphical check of the profiles
fig_check = True

# Name of the relevant columns in the files
cols_plants = {  # code or name of the associated end user
    'pod': cm.col_user,  # description of the associated end user
    'descrizione': cm.col_description,  # address of the end user
    'indirizzo': cm.col_address,  # size of the plant (kW)
    'pv_size': cm.col_size,  # annual energy produced (kWh)
    'produzione annua [kWh]': cm.col_energy,  # specific annual production (kWh/kWp)
    'rendita specifica [kWh/kWp]': cm.col_yield, }

cols_users = {  # code or name of the end user
    'pod': cm.col_user,  # description of the end user
    'descrizione': cm.col_description,  # address of the end user
    'indirizzo': cm.col_address,  # type of end user
    'tipo': cm.col_type,  # maximum available power of the end-user (kW)
    'potenza': cm.col_size, }

cols_user_bills = {'pod': cm.col_user,  # year
                   'anno': cm.col_year,  # number of the month
                   'mese': cm.col_month,  # ToU monthly consumption (kWh)
                   **{key: value for key, value in zip(('f0', 'f1', 'f2', 'f3'), cm.cols_tou_energy)},
                   'totale': cm.col_energy, }

col_production = 'Grid Export '  # 'Immissione in rete '  # hourly production of the plants (kW)

# ----------------------------------------------------------------------------
# Data loading

# Years calendar
f_year_col = lambda year, month, day, col: cm.df_year[
    ((cm.df_year[cm.col_year] == year) & (cm.df_year[cm.col_month] == month) & (cm.df_year[cm.col_day] == day))][col]

# Initialize empty datasets
data_users = DataFrame()  # list of the end users
data_plants = DataFrame()  # list of the plants
data_users_bills = DataFrame()  # list of the end user's bills
data_plants_year = DataFrame()  # list of the plants hourly production

# Get the list of municipality folders in the main directory
municipalities = os.listdir(directory_data)


def create_yearly_profile(df_plants_year, user_name=None):
    if user_name is None:
        user_name = user

    df_profile = profile.copy()

    if type(df_profile) != DataFrame:
        df_profile = DataFrame(df_profile,
                               columns=timedelta_range(start="0 Days", freq="1h", periods=df_profile.shape[1]))

    df_profile.index = date_range(start=f"{cm.ref_year}-01-01", end=f"{cm.ref_year}-12-31", freq="d")
    cols = [cm.col_year, cm.col_month, cm.col_day, cm.col_season, cm.col_week, cm.col_dayweek, cm.col_daytype]
    df_profile[cm.col_user] = user_name
    df_profile[cols] = cm.df_year.loc[df_profile.index, cols]
    df_plants_year = concat((df_plants_year, df_profile), axis=0)
    return df_profile, df_plants_year


path_municipality = ""
df_plants_year = DataFrame()

# Iterate over each municipality folder
for municipality in municipalities:
    # Create the complete path to the municipality folder
    path_municipality = os.path.join(directory_data, municipality)

    # Load and add the list of photovoltaic plants

    # TODO: create method for this line
    path_plants = os.path.join(path_municipality,
                               (file_plants if file_plants.endswith('.csv') else file_plants + '.csv'))

    # TODO: Drop unnecessary cols; maybe cols checking
    df_plants = read_csv(path_plants, sep=';').rename(columns=cols_plants)[
        cols_plants.values()]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_plants.insert(1, cm.col_municipality, municipality)  # add municipality
    data_plants = concat((data_plants, df_plants), axis=0)  # concatenate

    # Load and add the list of electric end user in the municipality
    path_users = os.path.join(path_municipality, (file_users if file_users.endswith('.csv') else file_users + '.csv'))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_users = read_csv(path_users, sep=';').rename(columns=cols_users)[cols_users.values()]
    df_users.insert(1, cm.col_municipality, municipality)  # add muncipality
    data_users = concat((data_users, df_users.dropna(axis=1)), axis=0)

    # Load monthly electricity consumption for each end user
    path_users_bills = os.path.join(path_municipality, (
        file_users_bills if file_users_bills.endswith('.csv') else file_users_bills + '.csv'))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_users_bills = read_csv(path_users_bills, sep=';').rename(columns=cols_user_bills)[cols_user_bills.values()]
    data_users_bills = concat((data_users_bills, df_users_bills), axis=0)  # concatenate

    # Load and add hourly electricity production for each plant

    # N.B. Here, more operations are required because:
    # - a single file must be opened for each plant
    # - the file contains multiple columns, only one is needed
    # - the file contains hourly data for one year, they must be
    # - reorganized by days config
    for user in df_plants[cm.col_user]:
        path_production = os.path.join(path_municipality, directory_production,
                                       (user if user.endswith('csv') else f'{user}.csv'))
        if directory_production == 'PVSOL':
            df_production = read_csv(path_production, sep=';', decimal=',', low_memory=False, skiprows=range(1, 17),
                                     index_col=0, header=0, parse_dates=True, date_format="%d.%m. %H:%M",
                                     usecols=["Time", col_production])
        elif directory_production == 'PVGIS':
            df_production = read_csv(path_production, sep=';', index_col=0, parse_dates=True,
                                     date_format="%d/%m/%Y %H:%M")
        else:
            raise ValueError()
        days = df_production[col_production].groupby(df_production.index.dayofyear)
        profile = DataFrame([items.values for g, items in days], index=days.groups.keys(),
                            columns=days.groups[1] - days.groups[1][0])
        # profile = reshape_array_by_year(profile, cm.ref_year)  # group by day
        df_profile, df_plants_year = create_yearly_profile(df_plants_year)

    data_plants_year = concat((data_plants_year, df_plants_year), axis=0)  # concatenate

# Reset indices after concatenation
for data in [data_plants, data_users, data_users_bills, data_plants_year]:
    data.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------------------------
# Check loaded data

# Check that each user has exactly 12 rows in the bills dataframe
assert np.all(data_users_bills[
                  cm.col_user].value_counts() == 12), "All end users in 'data_users_bills' must have exactly 12 rows."

# Check that all end users in the list and in the bill coincide
assert set(data_users[cm.col_user]) == set(
    data_users_bills[cm.col_user]), "All end users in 'data_users' must be also in 'data_users_bills."

# ----------------------------------------------------------------------------
# Manage the data

# Add column with total yearly consumption for each end user
data_users = data_users.merge(
    data_users_bills.groupby(cm.col_user)[cm.cols_tou_energy].sum().sum(axis=1).rename(cm.col_energy).reset_index(),
    on=cm.col_user)

# Add column with yearly consumption by ToU tariff for each end user
for col in cm.cols_tou_energy:
    data_users = data_users.merge(data_users_bills.groupby(cm.col_user)[col].sum().rename(col).reset_index(),
                                  on=cm.col_user)

# Change months in bills Dataframe from 0-11 to 1-12
if set(data_users_bills[cm.col_month]) == set(range(12)):
    data_users_bills[cm.col_month] = data_users_bills[cm.col_month].apply(lambda x: x + 1)

# ----------------------------------------------------------------------------
# Extract hourly consumption profiles from bills

data_users_year = DataFrame()  # year data of hourly profiles

# data_users_bs = data_users.merge(data_users_bills, on=col_user)
for user, data_bills in data_users_bills.groupby(cm.col_user):
    # type of pod (bta/ip/dom)
    pod_type = data_users.loc[data_users[cm.col_user] == user, cm.col_type].values[0]
    # pod_type = data_bills[col_type].iloc[0]
    # contractual power (kW)
    power = data_users.loc[data_users[cm.col_user] == user, cm.col_size].values[0]
    # power = data_bills[col_size].iloc[0]
    # type of bill (mono/tou)
    if not data_bills[cm.cols_tou_energy[1:]].isna().any().any():
        bills_cols = cm.cols_tou_energy[1:]
        bill_type = 'tou'
    else:
        if not data_bills[cm.cols_tou_energy[0]].isna().any():
            bills_cols = [cm.cols_tou_energy[0]]
        elif not data_bills[cm.col_energy].isna().any():
            bills_cols = [cm.col_energy]
        else:
            raise ValueError
        bill_type = 'mono'

    # Evaluate typical profiles in each month
    nds = []
    labels = []
    for _, df_b in data_bills.iterrows():
        month = df_b[cm.col_month]
        nds.append(cm.df_year[((cm.df_year[cm.col_year] == cm.ref_year) & (cm.df_year[cm.col_month] == month))].groupby(
            cm.col_daytype).count().iloc[:, 0].values)
        # extract labels to get yearly load profile
        labels.append(cm.df_year.loc[((cm.df_year[cm.col_year] == cm.ref_year) & (cm.df_year[cm.col_month] == month))][
                          cm.col_daytype].astype(int).values)

    nds = np.array(nds)
    bills = data_bills[bills_cols].values
    # scale typical profiles according to bills
    # TODO: option to profile estimation
    # in the future:
    # -function inputs can change
    # -different columns
    # -instead of monthly data, use daily data
    profiles = eval_profiles_gse(bills, nds, pod_type, bill_type)

    # Make yearly profile
    profile = []
    for il, label in enumerate(labels):
        profile.append(profiles[il].reshape(cm.nj, cm.ni)[label].flatten())
    profile = np.concatenate(profile)
    profile = reshape_array_by_year(profile, cm.ref_year)  # group by day
    df_profile, df_plants_year = create_yearly_profile(df_plants_year)

# ----------------------------------------------------------------------------
# %% Make yearly profile of families
# Load profiles of a single family
assert path_municipality != "", "'path_municipality' must be set."

bill = read_csv(os.path.join(path_municipality, 'bollette_domestici.csv'), sep=';', usecols=['f1', 'f2', 'f3'])
profiles = eval_profiles_gse(bill, cm.nds_ref, pod_type='dom', bill_type='tou')
profiles = profiles.reshape(cm.nj * cm.nm, cm.ni)

# Make yearly profile
profile = []
for label in cm.labels_ref:
    profile.append(profiles[label])
profile = np.concatenate(profile)
profile = reshape_array_by_year(profile, cm.ref_year)  # group by day
data_fam_year, df_plants_year = create_yearly_profile(df_plants_year, user_name="Dom")
# ----------------------------------------------------------------------------
# Evaluate "ToU" production and families consumption

data_plants_tou = DataFrame()
for user, df in data_plants_year.groupby(cm.col_user):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, cm.col_month], cm.ni)
    day_types = np.repeat(df.loc[:, cm.col_daytype], cm.ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((cm.nm, cm.nj * cm.ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([cm.col_month, cm.col_daytype]).count().iloc[:, 0].values.reshape(cm.nm, cm.nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((cm.nm, 1), np.nan), np.array(tou_energy)), axis=1)
    tou_energy = DataFrame(tou_energy, columns=cm.cols_tou_energy)
    tou_energy.insert(0, cm.col_user, user)
    tou_energy.insert(1, cm.col_year, cm.ref_year)
    tou_energy.insert(2, cm.col_month, cm.ms)
    # Concatenate
    data_plants_tou = concat((data_plants_tou, tou_energy), axis=0)

    # # check  # bills = [[0 for _ in fs] for _ in ms]  # for month, daytype, profiles in zip(df[col_month], df[col_daytype],  #                                     df.loc[:, 0:].values):  #     for if_, f in enumerate(fs):  #         bills[month-1][if_] += np.sum(profiles[arera[daytype]==f])  #  # bills = np.array(bills)

# Do the same for the families
data_fam_tou = DataFrame()
for user, df in data_fam_year.groupby(cm.col_user):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, cm.col_month], cm.ni)
    day_types = np.repeat(df.loc[:, cm.col_daytype], cm.ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((cm.nm, cm.nj * cm.ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([cm.col_month, cm.col_daytype]).count().iloc[:, 0].values.reshape(cm.nm, cm.nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((cm.nm, 1), np.nan), np.array(tou_energy)), axis=1)
    tou_energy = DataFrame(tou_energy, columns=cm.cols_tou_energy)
    tou_energy.insert(0, cm.col_user, user)
    tou_energy.insert(1, cm.col_year, cm.ref_year)
    tou_energy.insert(2, cm.col_month, cm.ms)
    # Concatenate
    data_fam_tou = concat((data_fam_tou, tou_energy), axis=0)

# ----------------------------------------------------------------------------
# Refinements

# Add column with yearly production by ToU tariff for each plant
for i, col in enumerate(cm.cols_tou_energy):
    if i == 0:
        data_plants[col] = np.nan
        continue
    data_plants = data_plants.merge(data_plants_tou.groupby(cm.col_user)[col].sum().rename(col).reset_index(),
                                    on=cm.col_user)

# Add column with type of plant !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if cm.col_type not in data_plants:
    data_plants[cm.col_type] = np.nan
data_plants[cm.col_type] = data_plants[cm.col_type].fillna('pv')

# ----------------------------------------------------------------------------
# %% Graphical check

if fig_check:

    # Families profiles
    # By month
    plt.figure()
    data = data_fam_year.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760, )
    for _, profile in data_fam_year.groupby(cm.col_user):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles, )
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()

    # Production profiles
    # By month
    plt.figure()
    data = data_plants_year.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760, )
    for _, profile in data_plants_year.groupby(cm.col_user):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles, )
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()

    # Consumption profiles
    for filter in ['bta', 'ip']:
        data = data_users_year.loc[
            data_users_year[cm.col_user].isin(data_users.loc[data_users[cm.col_type] == filter, cm.col_user])]

        # By month
        plt.figure()
        ddata = data.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
        for m, profile in ddata.iterrows():
            plt.plot(profile, label=str(m))
        plt.legend()
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()
        # Whole year
        plt.figure()
        profiles = np.zeros(8760, )
        for _, profile in data.groupby(cm.col_user):
            profiles += profile.loc[:, 0:].values.flatten()
        plt.plot(profiles, )
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()

        # Monthly consumption
        plt.figure()
        real = data_users.loc[data_users[cm.col_type] == filter].set_index([cm.col_user]).sort_index()[cm.col_energy]
        estim = data.groupby('user').sum().sort_index().loc[:, 0:].sum(axis=1)
        plt.barh(range(0, 2 * len(real), 2), real, label='Real')
        plt.barh(range(1, 1 + 2 * len(estim), 2), estim, label='Estimated')
        plt.legend()
        plt.yticks(range(0, 2 * len(real), 2), real.index)
        plt.xlabel('Energy, kWh')
        plt.title(f'Yearly consumption {filter.upper()}')
        plt.show()

# ----------------------------------------------------------------------------
# %% Save data

# Make output folder
os.makedirs(directory_out, exist_ok=True)

# Save all datasets
data_plants.to_csv(os.path.join(directory_out, "data_plants.csv"), sep=';', index=False)
data_users.to_csv(os.path.join(directory_out, "data_users.csv"), sep=';', index=False)
data_plants_tou.to_csv(os.path.join(directory_out, "data_plants_tou.csv"), sep=';', index=False)
data_users_bills.to_csv(os.path.join(directory_out, "data_users_tou.csv"), sep=';', index=False)
data_fam_tou.to_csv(os.path.join(directory_out, "data_fam_tou.csv"), sep=';', index=False)
data_users_year.to_csv(os.path.join(directory_out, "data_users_year.csv"), sep=';', index=False)
data_plants_year.to_csv(os.path.join(directory_out, "data_plants_year.csv"), sep=';', index=False)
data_fam_year.to_csv(os.path.join(directory_out, "data_fam_year.csv"), sep=';', index=False)
