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
from input.definitions import directory_data, directory_out, file_plants, directory_production, file_users, \
    file_users_bills, fig_check, cols_plants, cols_users, cols_user_bills, col_production
#
from utils import eval_x, eval_y_from_year

# ----------------------------------------------------------------------------


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
    data_plants[cm.col_type] = 'pv'
else:
    data_plants[cm.col_type] = data_plants[cm.col_type].fillna('pv')


# ----------------------------------------------------------------------------
# %% Graphical check

def vis_profiles():
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
    plt.close()


def by_month_profiles():
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
    plt.close()


def consumption_profiles():
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
        plt.close()

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
        plt.close()


if fig_check:
    vis_profiles()
    by_month_profiles()
    consumption_profiles()

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
