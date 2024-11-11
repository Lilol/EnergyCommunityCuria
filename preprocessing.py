# ----------------------------------------------------------------------------
# Import statement

# Data management

import numpy as np

from data_processing_pipeline.data_processing_arbiter import DataProcessingArbiter
from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from input.definitions import ColumnName, BillType, UserType
from input.reader import UsersReader, BillsReader, PvPlantReader, TariffReader, TypicalLoadProfileReader
from input.utility import reshape_array_by_year
from output.writer import Writer
from transform.definitions import create_profiles
from transform.extract.data_extractor import TariffExtractor
from transform.transform import TariffTransformer, TypicalLoadProfileTransformer
from utility import configuration
from utility.day_of_the_week import df_year
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("time_of_use_tariff", workers=(TariffReader(), TariffTransformer(), TariffExtractor())).execute()
DataProcessingPipeline("typical_load_profile",
                       workers=(TypicalLoadProfileReader(), TypicalLoadProfileTransformer())).execute()

arbiter = DataProcessingArbiter()

# Extract hourly consumption profiles from bills
year = configuration.config.getint("time", "year")
DataProcessingPipeline("pv_plant_profile", workers={})

data_users = UsersReader().execute()
data_users_bills = BillsReader().execute()

# merger = MyMerger()
# data_users_bs = merger.merge([data_users, data_users_bills], ColumnName.USER)

bills_cols = configuration.config.getarray("tariff", "time_of_use_labels", str)

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
    nds = np.zeros(shape=(1, len(data_bills)))
    day_types_over_the_year = []
    for i, (_, month) in enumerate(data_bills.loc[:, ColumnName.MONTH].iterrows()):
        # Number of days corresponding to this one in the year
        nds[i] = df_year.loc[df_year[ColumnName.MONTH] == month].groupby(ColumnName.DAY_TYPE).count().iloc[:, 0].values
        # Extract day types corresponding to a month
        day_types_over_the_year.append(
            df_year.loc[df_year[ColumnName.MONTH] == month, ColumnName.DAY_TYPE].astype(int).values)

    # TODO: option to profile estimation
    # scale typical profiles according to bills
    # in the future:
    # -function inputs can change
    # -different columns
    # -instead of monthly data, use daily data
    profiles = eval_profiles_gse(data_bills[bills_cols].values, nds, pod_type, bill_type)

    # Make yearly profile
    profile = []
    for il, label in enumerate(day_types_over_the_year):
        profile.append(profiles[il].reshape(nj, ni)[label].flatten())
    profile = np.concatenate(profile)
    profile = reshape_array_by_year(profile, year)  # group by day
    df_profile, df_plants_year = PvPlantReader.create_yearly_profile(df_plants_year)

# ----------------------------------------------------------------------------
# %% Make yearly profile of families
ni, nj, nm, ms = 0, 0, 0, 0

bill = BillsReader().execute()
profiles = eval_profiles_gse(bill, nds_ref, pod_type=UserType.PDMF, bill_type=BillType.TIME_OF_USE)
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
data_plants_tou = create_profiles(data_plants_year, ni, nj, nm, ms)
# Do the same for the families
data_fam_tou = create_profiles(data_fam_year, ni, nj, nm, ms)

# ----------------------------------------------------------------------------
# Refinements
# Add column with yearly production by ToU tariff for each plant

# ----------------------------------------------------------------------------
# %% Graphical check
if configuration.config.getboolean("visualization", "check_by_plotting"):
    vis_profiles(data_fam_year)
    by_month_profiles(data_plants_year)
    consumption_profiles(data_users)

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
