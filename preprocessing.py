# ----------------------------------------------------------------------------
# Import statement

# Data management
import os
import pandas as pd
import numpy as np

# Plotting
from matplotlib import pyplot as plt

#
from utils import eval_x, eval_y_from_year
from approach_gse import evaluate as eval_profiles_gse
#
from common import *


# ----------------------------------------------------------------------------
# Useful functions


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
cols_plants = {
    # code or name of the associated end user
    'pod': col_user,
    # description of the associated end user
    'descrizione': col_description,
    # address of the end user
    'indirizzo': col_address,
    # size of the plant (kW)
    'pv_size': col_size,
    # annual energy produced (kWh)
    'produzione annua [kWh]': col_energy,
    # specific annual production (kWh/kWp)
    'rendita specifica [kWh/kWp]': col_yield,
}
cols_users = {
    # code or name of the end user
    'pod': col_user,
    # description of the end user
    'descrizione': col_description,
    # address of the end user
    'indirizzo': col_address,
    # type of end user
    'tipo': col_type,
    # maximum available power of the end-user (kW)
    'potenza': col_size,
}

cols_user_bills = {
    'pod': col_user,
    # year
    'anno': col_year ,
    # number of the month
    'mese': col_month,
    # ToU monthly consumption (kWh)
    **{key: value for key, value in zip(['f0', 'f1', 'f2', 'f3'],
                                        cols_tou_energy)},
    'totale': col_energy,
}
col_production = 'Grid Export '  # 'Immissione in rete '  # hourly production of the plants (kW)

# ----------------------------------------------------------------------------
# Data loading

# Years calendar
f_year_col = lambda year, month, day, col: df_year[
    ((df_year[col_year] == year) & (df_year[col_month] == month) &
     (df_year[col_day] == day))][col]


# Initialize empty datasets
data_users = pd.DataFrame()  # list of the end users
data_plants = pd.DataFrame()  # list of the plants
data_users_bills = pd.DataFrame()  # list of the end user's bills
data_plants_year = pd.DataFrame()  # list of the plants hourly production

# Get the list of municipality folders in the main directory
municipalities = os.listdir(directory_data)

# Iterate over each municipality folder
for municipality in municipalities:
    # Create the complete path to the municipality folder
    path_municipality = os.path.join(directory_data, municipality)

    # Load and add the list of photovoltaic plants
    path_plants = \
        os.path.join(path_municipality,
                     (file_plants if file_plants.endswith('.csv')
                      else file_plants+'.csv'))
    df_plants = pd.read_csv(path_plants, sep=';').rename(columns=cols_plants)\
        [cols_plants.values()] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_plants.insert(1, col_municipality, municipality)  # add municipality
    data_plants = pd.concat((data_plants, df_plants), axis=0)  # concatenate

    # Load and add the list of electric end user in the municipality
    path_users = \
        os.path.join(path_municipality,
                     (file_users if file_users.endswith('.csv')
                      else file_users+'.csv'))
    df_users = pd.read_csv(path_users, sep=';').rename(columns=cols_users)\
        [cols_users.values()] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df_users.insert(1, col_municipality, municipality)  # add muncipality
    data_users = pd.concat((data_users, df_users.dropna(axis=1)), axis=0)

    # Load monthly electricity consumption for each end user
    path_users_bills = \
        os.path.join(path_municipality,
                     (file_users_bills if file_users_bills.endswith('.csv')
                      else file_users_bills+'.csv'))
    df_users_bills = pd.read_csv(path_users_bills, sep=';')\
        .rename(columns=cols_user_bills)[cols_user_bills.values()] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data_users_bills = \
        pd.concat((data_users_bills, df_users_bills), axis=0)  # concatenate

    # Load and add hourly electricity production for each plant

    # N.B. Here, more operations are required because:
    # - a single file must be opened for each plant
    # - the file contains multiple columns, only one is needed
    # - the file contains hourly data for one year, they must be
    # - reorganized by days
    df_plants_year = pd.DataFrame()
    for user in df_plants[col_user]:
        path_production = \
            os.path.join(path_municipality, directory_production,
                         (user if user.endswith('csv') else user+'.csv'))
        if directory_production == 'PVSOL':
            df_production = pd.read_csv(path_production,
                                        sep=';',
                                        decimal=',',
                                        low_memory=False)
            profile = \
                df_production.loc[14:, col_production]  # avoid rows of month data
            profile = profile.reset_index(drop=True)
            profile = \
                profile.str.replace(",", ".")  # right decimal
            profile = profile.astype(float).values #  from str to float
        elif directory_production == 'PVGIS':
            df_production = pd.read_csv(path_production, sep=';', index_col=0)
            profile = df_production.values
        else:
            raise ValueError()
        profile = reshape_array_by_year(profile, ref_year)  # group by day
        df_profile = pd.DataFrame(profile, index=range(1, 366))
        df_profile.insert(0, col_user, user)
        months = pd.to_datetime(df_profile.index, format='%j')\
            .strftime('%m').astype(int)
        df_profile.insert(1, col_year, ref_year)  # column with year
        df_profile.insert(2, col_month, months)  # column with month
        days = pd.to_datetime(df_profile.index, format='%j')\
            .strftime('%d').astype(int)
        df_profile.insert(3, col_day, days)  # column with day of the month
        df_profile.insert(2, col_season, None)  # column with season
        df_profile.insert(4, col_week, None)  # column with week
        df_profile.insert(6, col_dayweek, None)  # column with day of the week
        df_profile.insert(7, col_daytype, None)  # column with day type
        for i, df in df_profile.iterrows():
            df_profile.loc[i, col_season] = \
                f_year_col(df[col_year], df[col_month], df[col_day],
                           col_season) \
                    .values[0]
            df_profile.loc[i, col_week] = \
                f_year_col(df[col_year], df[col_month], df[col_day], col_week) \
                    .values[0]
            df_profile.loc[i, col_dayweek] = \
                f_year_col(df[col_year], df[col_month], df[col_day],
                           col_dayweek) \
                    .values[0]
            df_profile.loc[i, col_daytype] = \
                f_year_col(df[col_year], df[col_month], df[col_day],
                           col_daytype) \
                    .values[0]

        df_plants_year = pd.concat((df_plants_year, df_profile), axis=0)

    data_plants_year = \
        pd.concat((data_plants_year, df_plants_year), axis=0)  # concatenate

# Reset indices after concatenation
for data in [data_plants, data_users, data_users_bills, data_plants_year]:
    data.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------------------------
# Check loaded data

# Check that each user has exactly 12 rows in the bills dataframe
assert np.all(data_users_bills[col_user].value_counts() == 12), \
    "All end users in 'data_users_bills' must have exactly 12 rows."

# Check that all end users in the list and in the bill coincide
assert set(data_users[col_user]) == set(data_users_bills[col_user]), \
    "All end users in 'data_users' must be also in 'data_users_bills."

# ----------------------------------------------------------------------------
# Manage the data

# Add column with total yearly consumption for each end user
data_users = data_users.merge(
    data_users_bills.groupby(col_user)[cols_tou_energy]
    .sum().sum(axis=1).rename(col_energy).reset_index(), on=col_user)

# Add column with yearly consumption by ToU tariff for each end user
for col in cols_tou_energy:
    data_users = data_users.merge(
        data_users_bills.groupby(col_user)[col].sum().rename(col).reset_index(),
        on=col_user)

# Change months in bills Dataframe from 0-11 to 1-12
if set(data_users_bills[col_month]) == set(range(12)):
    data_users_bills[col_month] = \
        data_users_bills[col_month].apply(lambda x: x+1)

# ----------------------------------------------------------------------------
# Extract hourly consumption profiles from bills

data_users_year = pd.DataFrame()  # year data of hourly profiles

# data_users_bs = data_users.merge(data_users_bills, on=col_user)
for user, data_bills in data_users_bills.groupby(col_user):
    # type of pod (bta/ip/dom)
    pod_type = data_users.loc[data_users[col_user] == user, col_type].values[0]
    # pod_type = data_bills[col_type].iloc[0]
    # contractual power (kW)
    power = data_users.loc[data_users[col_user] == user, col_size].values[0]
    # power = data_bills[col_size].iloc[0]
    # type of bill (mono/tou)
    if not data_bills[cols_tou_energy[1:]].isna().any().any():
        bills_cols = cols_tou_energy[1:]
        bill_type = 'tou'
    else:
        if not data_bills[cols_tou_energy[0]].isna().any():
            bills_cols = [cols_tou_energy[0]]
        elif not data_bills[col_energy].isna().any():
            bills_cols = [col_energy]
        else:
            raise ValueError
        bill_type = 'mono'

    # Evaluate typical profiles in each month
    nds = []
    labels = []
    for _, df_b in data_bills.iterrows():
        month = df_b[col_month]
        nds.append(
            df_year[((df_year[col_year] == ref_year) &
                     (df_year[col_month] == month))]\
                .groupby(col_daytype).count().iloc[:, 0].values)
        # extract labels to get yearly load profile
        labels.append(df_year.loc[((df_year[col_year] == ref_year) &
                               (df_year[col_month] == month))][col_daytype]
                      .astype(int).values)

    nds = np.array(nds)
    bills = data_bills[bills_cols].values
    profiles = eval_profiles_gse(bills, nds, pod_type, bill_type)

    # Make yearly profile
    profile = []
    for il, label in enumerate(labels):
        profile.append(profiles[il].reshape(nj, ni)[label].flatten())
    profile = np.concatenate(profile)
    profile = reshape_array_by_year(profile, ref_year)  # group by day
    df_profile = pd.DataFrame(profile, index=range(1, 366))
    df_profile.insert(0, col_user, user)
    months = pd.to_datetime(df_profile.index, format='%j') \
        .strftime('%m').astype(int)
    df_profile.insert(1, col_year, ref_year)  # column with year
    df_profile.insert(2, col_month, months)  # column with month
    days = pd.to_datetime(df_profile.index, format='%j') \
        .strftime('%d').astype(int)
    df_profile.insert(3, col_day, days)  # column with day of the month
    df_profile.insert(2, col_season, None)  # column with season
    df_profile.insert(4, col_week, None)  # column with week
    df_profile.insert(6, col_dayweek, None)  # column with day of the week
    df_profile.insert(7, col_daytype, None)  # column with day type
    for i, df in df_profile.iterrows():
        df_profile.loc[i, col_season] = \
            f_year_col(df[col_year], df[col_month], df[col_day], col_season)\
                .values[0]
        df_profile.loc[i, col_week] = \
            f_year_col(df[col_year], df[col_month], df[col_day], col_week)\
                .values[0]
        df_profile.loc[i, col_dayweek] = \
            f_year_col(df[col_year], df[col_month], df[col_day], col_dayweek)\
                .values[0]
        df_profile.loc[i, col_daytype] = \
            f_year_col(df[col_year], df[col_month], df[col_day], col_daytype)\
                .values[0]

    # Concatenate
    data_users_year = pd.concat((data_users_year, df_profile), axis=0)

# ----------------------------------------------------------------------------
#%% Make yearly profile of families
# Load profiles of a single family
bill = pd.read_csv(os.path.join(path_municipality, 'bollette_domestici.csv'),
                   sep=';')[['f1', 'f2', 'f3']].values
profiles = eval_profiles_gse(bill, nds_ref, pod_type='dom', bill_type='tou')
profiles = profiles.reshape(nj*nm, ni)

# Make yearly profile
profile = []
for label in labels_ref:
    profile.append(profiles[label])
profile = np.concatenate(profile)
profile = reshape_array_by_year(profile, ref_year)  # group by day

data_fam_year = pd.DataFrame(profile, index=range(1, 366))
data_fam_year.insert(0, col_user, 'Dom')
months = pd.to_datetime(data_fam_year.index, format='%j') \
    .strftime('%m').astype(int)
data_fam_year.insert(1, col_year, ref_year)  # column with year
data_fam_year.insert(2, col_month, months)  # column with month
days = pd.to_datetime(data_fam_year.index, format='%j') \
    .strftime('%d').astype(int)
data_fam_year.insert(3, col_day, days)  # column with day of the month
data_fam_year.insert(2, col_season, None)  # column with season
data_fam_year.insert(4, col_week, None)  # column with week
data_fam_year.insert(6, col_dayweek, None)  # column with day of the week
data_fam_year.insert(7, col_daytype, None)  # column with day type
for i, df in data_fam_year.iterrows():
    data_fam_year.loc[i, col_season] = \
        f_year_col(df[col_year], df[col_month], df[col_day], col_season) \
            .values[0]
    data_fam_year.loc[i, col_week] = \
        f_year_col(df[col_year], df[col_month], df[col_day], col_week) \
            .values[0]
    data_fam_year.loc[i, col_dayweek] = \
        f_year_col(df[col_year], df[col_month], df[col_day], col_dayweek) \
            .values[0]
    data_fam_year.loc[i, col_daytype] = \
        f_year_col(df[col_year], df[col_month], df[col_day], col_daytype) \
            .values[0]


# ----------------------------------------------------------------------------
# Evaluate "ToU" production and families consumption

data_plants_tou = pd.DataFrame()
for user, df in data_plants_year.groupby(col_user):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, col_month], ni)
    day_types = np.repeat(df.loc[:, col_daytype], ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((nm, nj*ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([col_month, col_daytype]).count().iloc[:, 0].values\
        .reshape(nm, nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)),
                                axis=1)
    tou_energy = pd.DataFrame(tou_energy, columns=cols_tou_energy)
    tou_energy.insert(0, col_user, user)
    tou_energy.insert(1, col_year, ref_year)
    tou_energy.insert(2, col_month, ms)
    # Concatenate
    data_plants_tou = pd.concat((data_plants_tou, tou_energy), axis=0)

    # # check
    # bills = [[0 for _ in fs] for _ in ms]
    # for month, daytype, profiles in zip(df[col_month], df[col_daytype],
    #                                     df.loc[:, 0:].values):
    #     for if_, f in enumerate(fs):
    #         bills[month-1][if_] += np.sum(profiles[arera[daytype]==f])
    #
    # bills = np.array(bills)

# Do the same for the families
data_fam_tou = pd.DataFrame()
for user, df in data_fam_year.groupby(col_user):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, col_month], ni)
    day_types = np.repeat(df.loc[:, col_daytype], ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((nm, nj*ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([col_month, col_daytype]).count().iloc[:, 0].values\
        .reshape(nm, nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)),
                                axis=1)
    tou_energy = pd.DataFrame(tou_energy, columns=cols_tou_energy)
    tou_energy.insert(0, col_user, user)
    tou_energy.insert(1, col_year, ref_year)
    tou_energy.insert(2, col_month, ms)
    # Concatenate
    data_fam_tou = pd.concat((data_fam_tou, tou_energy), axis=0)

# ----------------------------------------------------------------------------
# Refinements

# Add column with yearly production by ToU tariff for each plant
for i, col in enumerate(cols_tou_energy):
    if i == 0:
        data_plants[col] = np.nan
        continue
    data_plants = data_plants.merge(
        data_plants_tou.groupby(col_user)[col].sum().rename(col).reset_index(),
        on=col_user)

# Add column with type of plant !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if col_type not in data_plants:
    data_plants[col_type] = np.nan
data_plants[col_type] = data_plants[col_type].fillna('pv')

# ----------------------------------------------------------------------------
#%% Graphical check

if fig_check:

    # Families profiles
    # By month
    plt.figure()
    data = data_fam_year.groupby(['user', 'month']).mean()\
        .groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760,)
    for _, profile in data_fam_year.groupby(col_user):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles,)
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()

    # Production profiles
    # By month
    plt.figure()
    data = data_plants_year.groupby(['user', 'month']).mean()\
        .groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760,)
    for _, profile in data_plants_year.groupby(col_user):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles,)
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()

    # Consumption profiles
    for filter in ['bta', 'ip']:
        data = data_users_year.loc[
            data_users_year[col_user].isin(
                data_users.loc[data_users[col_type]==filter, col_user])]

        # By month
        plt.figure()
        ddata = data.groupby(['user', 'month']).mean().groupby('month').sum()\
                    .loc[:, 0:]
        for m, profile in ddata.iterrows():
            plt.plot(profile, label=str(m))
        plt.legend()
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()
        # Whole year
        plt.figure()
        profiles = np.zeros(8760,)
        for _, profile in data.groupby(col_user):
            profiles += profile.loc[:, 0:].values.flatten()
        plt.plot(profiles,)
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()

        # Monthly consumption
        plt.figure()
        real = data_users.loc[data_users[col_type] == filter]\
            .set_index([col_user]).sort_index()[col_energy]
        estim = data.groupby('user').sum().sort_index().loc[:, 0:].sum(axis=1)
        plt.barh(range(0, 2*len(real), 2), real, label='Real')
        plt.barh(range(1, 1+2*len(estim), 2), estim, label='Estimated')
        plt.legend()
        plt.yticks(range(0, 2*len(real), 2), real.index)
        plt.xlabel('Energy, kWh')
        plt.title(f'Yearly consumption {filter.upper()}')
        plt.show()

# ----------------------------------------------------------------------------
#%% Save data

# Make output folder
os.makedirs(directory_out , exist_ok=True)

# Save all datasets
data_plants.to_csv(os.path.join(directory_out, "data_plants.csv"),
                   sep=';', index=False)
data_users.to_csv(os.path.join(directory_out, "data_users.csv"),
                  sep=';', index=False)
data_plants_tou.to_csv(os.path.join(directory_out, "data_plants_tou.csv"),
                        sep=';', index=False)
data_users_bills.to_csv(os.path.join(directory_out, "data_users_tou.csv"),
                        sep=';', index=False)
data_fam_tou.to_csv(os.path.join(directory_out, "data_fam_tou.csv"),
                        sep=';', index=False)
data_users_year.to_csv(os.path.join(directory_out, "data_users_year.csv"),
                       sep=';', index=False)
data_plants_year.to_csv(os.path.join(directory_out, "data_plants_year.csv"),
                        sep=';', index=False)
data_fam_year.to_csv(os.path.join(directory_out, "data_fam_year.csv"),
                       sep=';', index=False)