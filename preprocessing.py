# ----------------------------------------------------------------------------
# Import statement

# Data management

import numpy as np
# Plotting
from pandas import DataFrame, concat

#
import configuration
from input.definitions import ColumnName, BillType
from input.reader import UsersReader, BillsReader, PvPlantReader
from input.utility import reshape_array_by_year
from output.writer import Writer
from time.day_of_the_week import df_year
from transform.approach_gse import evaluate as eval_profiles_gse
from transform.merger import MyMerger
from utils import eval_x, eval_y_from_year
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles

# ----------------------------------------------------------------------------
# Extract hourly consumption profiles from bills
year = configuration.config.getint("time", "year")
data_users = UsersReader().execute()
data_users_bills = BillsReader().execute()

merger = MyMerger()
data_users_bs = merger.merge([data_users, data_users_bills], ColumnName.USER)

for user, data_bills in data_users_bills.groupby(ColumnName.USER):
    # type of pod (bta/ip/dom)
    pod_type = data_users.loc[user, ColumnName.USER_TYPE]
    # pod_type = data_bills[ColumnName.USER_TYPE].iloc[0]
    # type of bill (mono/tou)
    if not data_bills[BillsReader.time_of_use_labels[1:]].isna().any(how=None):
        bills_cols = BillsReader.time_of_use_labels[1:]
        bill_type = BillType.TIME_OF_USE
    else:
        if not data_bills[BillsReader.time_of_use_labels[0]].isna().any():
            bills_cols = [BillsReader.time_of_use_labels[0]]
        elif not data_bills[BillsReader.time_of_use_labels].isna().any():
            bills_cols = [BillsReader.time_of_use_labels]
        else:
            raise ValueError
        bill_type = BillType.MONO

    # Evaluate typical profiles in each month
    nds = []
    labels = []
    for _, month in data_bills.loc[:, ColumnName.MONTH].iterrows():
        # Number of days corresponding to this one in the year
        nds.append(df_year[((df_year[ColumnName.YEAR] == year) & (df_year[ColumnName.MONTH] == month))].groupby(
            ColumnName.DAY_TYPE).count().iloc[:, 0].values)
        # Extract labels to get yearly load profile
        labels.append(df_year.loc[((df_year[ColumnName.YEAR] == year) & (df_year[ColumnName.MONTH] == month))][
                          ColumnName.DAY_TYPE].astype(int).values)

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
        profile.append(profiles[il].reshape(nj, ni)[label].flatten())
    profile = np.concatenate(profile)
    profile = reshape_array_by_year(profile, year)  # group by day
    df_profile, df_plants_year = PvPlantReader.create_yearly_profile(df_plants_year)

# ----------------------------------------------------------------------------
# %% Make yearly profile of families

bill = BillsReader().execute()
profiles = eval_profiles_gse(bill, nds_ref, pod_type='dom', bill_type='tou')
profiles = profiles.reshape(nj * nm, ni)

# Make yearly profile
profile = []
for label in labels_ref:
    profile.append(profiles[label])
profile = np.concatenate(profile)
profile = reshape_array_by_year(profile, year)  # group by day
data_fam_year, df_plants_year = PvPlantReader.create_yearly_profile(df_plants_year, user_name="Dom")

# ----------------------------------------------------------------------------
# Evaluate "ToU" production and families consumption
data_plants_tou = DataFrame()
for user, df in data_plants_year.groupby(ColumnName.USER):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, ColumnName.MONTH], ni)
    day_types = np.repeat(df.loc[:, ColumnName.DAY_TYPE], ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((nm, nj * ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([ColumnName.MONTH, ColumnName.DAY_TYPE]).count().iloc[:, 0].values.reshape(nm, nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)), axis=1)
    tou_energy = DataFrame(tou_energy, columns=BillsReader.time_of_use_energy_column_names)
    tou_energy.insert(0, ColumnName.USER, user)
    tou_energy.insert(1, ColumnName.YEAR, year)
    tou_energy.insert(2, ColumnName.MONTH, ms)
    # Concatenate
    data_plants_tou = concat((data_plants_tou, tou_energy), axis=0)

    # # check  # bills = [[0 for _ in fs] for _ in ms]  # for month, daytype, profiles in zip(df[ColumnName.MONTH], df[ColumnName.DAY_TYPE],  #                                     df.loc[:, 0:].values):  #     for if_, f in enumerate(fs):  #         bills[month-1][if_] += np.sum(profiles[arera[daytype]==f])  #  # bills = np.array(bills)

# Do the same for the families
data_fam_tou = DataFrame()
for user, df in data_fam_year.groupby(ColumnName.USER):
    # Evaluate profiles in typical days
    months = np.repeat(df.loc[:, ColumnName.MONTH], ni)
    day_types = np.repeat(df.loc[:, ColumnName.DAY_TYPE], ni)
    profiles = df.loc[:, 0:].values.flatten()
    profiles = eval_y_from_year(profiles, months, day_types).reshape((nm, nj * ni))
    # Evaluate typical profiles in each month
    nds = df.groupby([ColumnName.MONTH, ColumnName.DAY_TYPE]).count().iloc[:, 0].values.reshape(nm, nj)
    tou_energy = []
    for y, nd in zip(profiles, nds):
        tou_energy.append(eval_x(y, nd))
    # Create dataframe
    tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)), axis=1)
    tou_energy = DataFrame(tou_energy, columns=BillsReader.time_of_use_energy_column_names)
    tou_energy.insert(0, ColumnName.USER, user)
    tou_energy.insert(1, ColumnName.YEAR, year)
    tou_energy.insert(2, ColumnName.MONTH, ms)
    # Concatenate
    data_fam_tou = concat((data_fam_tou, tou_energy), axis=0)

# ----------------------------------------------------------------------------
# Refinements

# Add column with yearly production by ToU tariff for each plant
for i, col in enumerate(BillsReader.time_of_use_energy_column_names):
    if i == 0:
        data_plants[col] = np.nan
        continue
    data_plants = data_plants.merge(data_plants_tou.groupby(ColumnName.USER)[col].sum().rename(col).reset_index(),
                                    on=ColumnName.USER)

# Add column with type of plant
data_plants[ColumnName.USER_TYPE] = 'pv' if ColumnName.USER_TYPE not in data_plants else data_plants[
    ColumnName.USER_TYPE].fillna('pv')

# ----------------------------------------------------------------------------
# %% Graphical check
if configuration.config.getboolean("visualization", "check_by_plotting"):
    vis_profiles()
    by_month_profiles()
    consumption_profiles()

# ----------------------------------------------------------------------------
# %% Save data
data_writer = Writer()
data_writer.write(data_plants, "data_plants")
data_writer.write(data_users, "data_users")
data_writer.write(data_plants_tou, "data_plants_tou")
data_writer.write(data_users_bills, "data_users_bills")
data_writer.write(data_fam_tou, "data_fam_tou")
data_writer.write(data_users_year, "data_users_year")
data_writer.write(data_plants_year, "data_plants_year")
data_writer.write(data_fam_year, "data_fam_year")
