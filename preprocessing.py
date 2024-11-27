from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.store_data import Store
from input.reader import UsersReader, BillReader, PvPlantReader, TariffReader, TypicalLoadProfileReader, \
    ProductionReader
from output.write import Write
from transform.combine.combine import TypicalMonthlyConsumptionCalculator, YearlyConsumptionCombiner
from transform.extract.data_extractor import TariffExtractor, TouExtractor, DayTypeExtractor, DayCountExtractor, \
    TypicalYearExtractor
from transform.transform import TariffTransformer, TypicalLoadProfileTransformer, UserDataTransformer, \
    PvPlantDataTransformer, BillDataTransformer, ProductionDataTransformer, BillLoadProfileTransformer, \
    YearlyProfileCreator, ProfileDataAggregator
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import vis_profiles, by_month_profiles, consumption_profiles
from visualization.visualize import Visualize

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("time_of_use", workers=(
    TariffReader(),
    TariffTransformer(),
    TariffExtractor(),
    Store("time_of_use_time_slots"),
    TouExtractor(),
    Store("time_of_use_tariff"))).execute()


DataProcessingPipeline("day_properties", workers=(
    DayTypeExtractor(),
    Store("day_types"),
    DayCountExtractor(),
    Store("day_count"))).execute()


DataProcessingPipeline("typical_load_profile", workers=(
    TypicalLoadProfileReader(),
    TypicalLoadProfileTransformer(),
    Store("typical_load_profiles_gse"),
    TypicalMonthlyConsumptionCalculator(),
    Store("typical_aggregated_consumption"))).execute()


DataProcessingPipeline("load_profiles_from_bills", workers=(
    BillReader(),
    BillDataTransformer(),
    Store("bills"),
    Write("data_users_bills"),
    BillLoadProfileTransformer(),
    Store("load_profiles_from_bills"),
    YearlyProfileCreator(),
    Store("yearly_load_profiles_from_bills"),
    Write("data_users_years"))).execute()


DataProcessingPipeline("users", workers=(
    UsersReader(),
    UserDataTransformer(),
    YearlyConsumptionCombiner(),
    Store("users"),
    Write("data_users"),
    Visualize("consumption_profiles", consumption_profiles))).execute()


DataProcessingPipeline("families", workers=(
    BillReader(filename="bollette_domestici.csv"),
    BillDataTransformer(),
    Store("families_bills"),
    BillLoadProfileTransformer(),
    YearlyProfileCreator(),
    Store("yearly_load_profiles_families"),
    Write("data_families_years"))).execute()


DataProcessingPipeline("pv_plants", workers=(
    PvPlantReader(),
    PvPlantDataTransformer(),
    Store("pv_plants"),
    Write("data_plants"),
    Visualize("profiles", vis_profiles),
    ProfileDataAggregator(),
    Write("data_plants_tou"))).execute()


DataProcessingPipeline("pv_production", workers=(
    ProductionReader(),
    ProductionDataTransformer(),
    TypicalYearExtractor(),
    Store("pv_profiles"),
    YearlyProfileCreator(),
    Write("data_plants_year"),
    Visualize("profiles_by_month", by_month_profiles))).execute()
