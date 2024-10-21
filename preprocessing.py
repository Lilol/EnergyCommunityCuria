# ----------------------------------------------------------------------------
# Import statement

# Data management
import os

import numpy as np
# Plotting
from pandas import DataFrame, concat, read_csv

#
import common as cm
import configuration
from approach_gse import evaluate as eval_profiles_gse
from input.definitions import InputColumn

from utils import eval_x, eval_y_from_year
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles


# ----------------------------------------------------------------------------
# Extract hourly consumption profiles from bills

data_users_year = DataFrame()  # year data of hourly profiles

# data_users_bs = data_users.merge(data_users_bills, on=col_user)
for user, data_bills in data_users_bills.groupby(InputColumn.USER):
    # type of pod (bta/ip/dom)
    pod_type = data_users.loc[data_users[InputColumn.USER] == user, cm.col_type].values[0]
    # pod_type = data_bills[col_type].iloc[0]
    # contractual power (kW)
    power = data_users.loc[data_users[InputColumn.USER] == user, cm.col_size].values[0]
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
    # power = data_bills[col_size].iloc[0]

    # Evaluate typical profiles in each month
    nds = []
    labels = []
    for _, month in data_bills.loc[:, InputColumn.MONTH].iterrows():
        # Number of days corresponding to this one in the year
        nds.append(cm.df_year[((cm.df_year[cm.col_year] == cm.ref_year) & (cm.df_year[cm.col_month] == month))].groupby(
            cm.col_daytype).count().iloc[:, 0].values)
        # Extract labels to get yearly load profile
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
for user, df in data_plants_year.groupby(InputColumn.USER):
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
    tou_energy.insert(0, InputColumn.USER, user)
    tou_energy.insert(1, InputColumn.YEAR, cm.ref_year)
    tou_energy.insert(2, InputColumn.MONTH, cm.ms)
    # Concatenate
    data_plants_tou = concat((data_plants_tou, tou_energy), axis=0)

    # # check  # bills = [[0 for _ in fs] for _ in ms]  # for month, daytype, profiles in zip(df[col_month], df[col_daytype],  #                                     df.loc[:, 0:].values):  #     for if_, f in enumerate(fs):  #         bills[month-1][if_] += np.sum(profiles[arera[daytype]==f])  #  # bills = np.array(bills)

# Do the same for the families
data_fam_tou = DataFrame()
for user, df in data_fam_year.groupby(InputColumn.USER):
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
    tou_energy.insert(0, InputColumn.USER, user)
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
    data_plants = data_plants.merge(data_plants_tou.groupby(InputColumn.USER)[col].sum().rename(col).reset_index(),
                                    on=InputColumn.USER)

# Add column with type of plant
if cm.col_type not in data_plants:
    data_plants[cm.col_type] = 'pv'
else:
    data_plants[cm.col_type] = data_plants[cm.col_type].fillna('pv')

# ----------------------------------------------------------------------------
# %% Graphical check
if configuration.config.getboolean("visualization", "check_by_plotting"):
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
