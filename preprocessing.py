import numpy as np

from data_processing_pipeline.data_processing_arbiter import DataProcessingArbiter
from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import StoreData
from input.definitions import BillType, UserType
from input.reader import UsersReader, BillReader, PvPlantReader, TariffReader, TypicalLoadProfileReader, \
    ProductionReader
from input.utility import reshape_array_by_year
from output.writer import Writer
from transform.combine.approach_gse import evaluate
from transform.combine.combine import TypicalMonthlyConsumptionCalculator
from transform.definitions import create_profiles
from transform.extract.data_extractor import TariffExtractor, TouExtractor, DayTypeExtractor, DayCountExtractor, \
    TypicalMeteorologicalYearExtractor
from transform.extract.utils import ProfileExtractor
from transform.transform import TariffTransformer, TypicalLoadProfileTransformer, UserDataTransformer, \
    PvPlantDataTransformer, BillDataTransformer, ProductionDataTransformer, BillLoadProfileTransformer, \
    PvProfileTransformer
from utility import configuration
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("time_of_use_tariff", workers=(
    TariffReader(), TariffTransformer(), TariffExtractor(), StoreData("time_of_use_time_slots"),
    TouExtractor())).execute()
DataProcessingPipeline("day_count", workers=(DayTypeExtractor(), StoreData("day_types"), DayCountExtractor())).execute()
DataProcessingPipeline("typical_load_profile_gse",
                       workers=(TypicalLoadProfileReader(), TypicalLoadProfileTransformer())).execute()
DataProcessingPipeline("typical_aggregated_consumption", workers=(TypicalMonthlyConsumptionCalculator(),)).execute()

DataProcessingPipeline("users", workers=(UsersReader(), UserDataTransformer())).execute()
DataProcessingPipeline("bills", workers=(BillReader(), BillDataTransformer())).execute()
DataProcessingPipeline("pv_plants", workers=(PvPlantReader(), PvPlantDataTransformer())).execute()
DataProcessingPipeline("pv_production", workers=(
ProductionReader(), ProductionDataTransformer(), TypicalMeteorologicalYearExtractor())).execute()

DataProcessingPipeline("load_profiles_from_bills", workers=(BillLoadProfileTransformer(),)).execute()
DataProcessingPipeline("pv_profile", workers=(PvProfileTransformer(),)).execute()

arbiter = DataProcessingArbiter()
data_store = DataStore()

# ----------------------------------------------------------------------------
# %% Make yearly profile of families
ni, nj, nm, ms = 0, 0, 0, 0

bill = BillReader().execute()
profiles = evaluate(bill, nds_ref, pod_type=UserType.PDMF, bill_type=BillType.TIME_OF_USE)
profiles = profiles.reshape(nj * nm, ni)

# Make yearly profile
profile = []
for label in labels_ref:
    profile.append(profiles[label])
profile = np.concatenate(profile)
profile = reshape_array_by_year(profile, year)  # group by day
data_fam_year, df_plants_year = ProfileExtractor.create_typical_profile_from_yearly_profile(df_plants_year)

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
data_writer = Writer("writer")
data_writer.write(data_plants, "data_plants")
data_writer.write(data_users, "data_users")
data_writer.write(data_plants_tou, "data_plants_tou")
data_writer.write(data_users_bills, "data_users_bills")
data_writer.write(data_fam_tou, "data_fam_tou")
data_writer.write(data_users_year, "data_users_year")
data_writer.write(data_plants_year, "data_plants_year")
data_writer.write(data_fam_year, "data_fam_year")
