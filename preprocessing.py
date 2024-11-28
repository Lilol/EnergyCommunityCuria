from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.store_data import Store
from input.read import ReadUserData, ReadBills, ReadPvPlantData, ReadTariff, ReadTypicalLoadProfile, ReadProduction
from output.write import Write
from transform.combine.combine import CalculateTypicalMonthlyConsumption, AddYearlyConsumptionToBillData
from transform.extract.data_extractor import ExtractTimeOfUseParameters, ExtractDayTypesInTimeframe, \
    ExtractDayCountInTimeframe, ExtractTypicalYear
from transform.transform import TransformTariffData, TypicalLoadProfileTransformer, TransformUserData, \
    TransformPvPlantData, TransformBills, TransformProduction, TransformBillsToLoadProfiles, CreateYearlyProfile, \
    AggregateProfileDataForTimePeriod, TransformTimeOfUseTimeSlots
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import plot_family_profiles, plot_pv_profiles, plot_consumption_profiles
from visualization.visualize import Visualize

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("time_of_use", workers=(
    ReadTariff(),
    TransformTariffData(),
    ExtractTimeOfUseParameters(),
    Store("time_of_use_time_slots"),
    TransformTimeOfUseTimeSlots(),
    Store("time_of_use_tariff"))).execute()


DataProcessingPipeline("day_properties", workers=(
    ExtractDayTypesInTimeframe(),
    Store("day_types"),
    ExtractDayCountInTimeframe(),
    Store("day_count"))).execute()


DataProcessingPipeline("typical_load_profile", workers=(
    ReadTypicalLoadProfile(),
    TypicalLoadProfileTransformer(),
    Store("typical_load_profiles_gse"),
    CalculateTypicalMonthlyConsumption(),
    Store("typical_aggregated_consumption"))).execute()


DataProcessingPipeline("load_profiles_from_bills", workers=(
    ReadBills(),
    TransformBills(),
    Store("bills"),
    Write("data_users_bills"),
    TransformBillsToLoadProfiles(),
    Store("load_profiles_from_bills"),
    CreateYearlyProfile(),
    Store("yearly_load_profiles_from_bills"),
    Write("data_users_years"))).execute()


DataProcessingPipeline("users", workers=(
    ReadUserData(),
    TransformUserData(),
    AddYearlyConsumptionToBillData(),
    Store("users"),
    Write("data_users"),
    Visualize("consumption_profiles", plot_consumption_profiles))).execute()


DataProcessingPipeline("families", workers=(
    ReadBills(filename="bollette_domestici.csv"),
    TransformBills(),
    Store("families_bills"),
    TransformBillsToLoadProfiles(),
    CreateYearlyProfile(),
    Store("yearly_load_profiles_families"),
    Write("data_families_year"),
    Visualize("profiles", plot_family_profiles))).execute()


DataProcessingPipeline("pv_plants", workers=(
    ReadPvPlantData(),
    TransformPvPlantData(),
    Store("pv_plants"),
    Write("data_plants"),
    Write("data_plants_tou"))).execute()


DataProcessingPipeline("pv_production", workers=(
    ReadProduction(),
    TransformProduction(),
    ExtractTypicalYear(),
    Store("pv_profiles"),
    CreateYearlyProfile(),
    AggregateProfileDataForTimePeriod(),
    Write("data_plants_year"),
    Visualize("profiles_by_month", plot_pv_profiles))).execute()
