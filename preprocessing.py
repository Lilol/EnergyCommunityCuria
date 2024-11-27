from data_processing_pipeline.data_processing_arbiter import DataProcessingArbiter
from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from input.reader import UsersReader, BillReader, PvPlantReader, TariffReader, TypicalLoadProfileReader, \
    ProductionReader
from output.writer import Writer
from transform.combine.combine import TypicalMonthlyConsumptionCalculator, YearlyConsumptionCombiner
from transform.extract.data_extractor import TariffExtractor, TouExtractor, DayTypeExtractor, DayCountExtractor, \
    TypicalYearExtractor
from transform.transform import TariffTransformer, TypicalLoadProfileTransformer, UserDataTransformer, \
    PvPlantDataTransformer, BillDataTransformer, ProductionDataTransformer, BillLoadProfileTransformer, \
    YearlyProfileCreator
from utility import configuration
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("time_of_use", workers=(
    TariffReader(), TariffTransformer(), TariffExtractor(), Store("time_of_use_time_slots"), TouExtractor(),
    Store("time_of_use_tariff"))).execute()
DataProcessingPipeline("day_properties", workers=(
    DayTypeExtractor(), Store("day_types"), DayCountExtractor(), Store("day_count"))).execute()
DataProcessingPipeline("typical_load_profile", workers=(
    TypicalLoadProfileReader(), TypicalLoadProfileTransformer(), Store("typical_load_profiles_gse"),
    TypicalMonthlyConsumptionCalculator(), Store("typical_aggregated_consumption"))).execute()

DataProcessingPipeline("load_profiles_from_bills", workers=(
    BillReader(), BillDataTransformer(), Store("bills"), Writer("data_users_bills"), BillLoadProfileTransformer(),
    Store("load_profiles_from_bills"), YearlyProfileCreator(), Store("yearly_load_profiles_from_bills"),
    Writer("data_users_years"))).execute()

DataProcessingPipeline("users", workers=(
    UsersReader(), UserDataTransformer(), YearlyConsumptionCombiner(), Store("users"), Writer("data_users"))).execute()

DataProcessingPipeline("pv_plants", workers=(PvPlantReader(), PvPlantDataTransformer(), Store("pv_plants"))).execute()
DataProcessingPipeline("pv_production", workers=(
    ProductionReader(), ProductionDataTransformer(), TypicalYearExtractor(), Store("pv_profiles"),
    YearlyProfileCreator(), Writer("data_plants_year"))).execute()

arbiter = DataProcessingArbiter()
data_store = DataStore()

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
