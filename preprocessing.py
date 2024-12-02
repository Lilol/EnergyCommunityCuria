from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from input.definitions import UserType
from input.read import ReadUserData, ReadBills, ReadPvPlantData, ReadTariff, ReadTypicalLoadProfile, ReadProduction
from output.write import Write, WriteSeparately
from transform.check import CheckAnnualSum
from transform.combine.combine import CalculateTypicalMonthlyConsumption, AddYearlyConsumptionToBillData
from transform.extract.data_extractor import ExtractTimeOfUseParameters, ExtractDayTypesInTimeframe, \
    ExtractDayCountInTimeframe, ExtractTypicalYear
from transform.transform import TransformTariffData, TransformTypicalLoadProfile, TransformUserData, \
    TransformPvPlantData, TransformBills, TransformProduction, TransformBillsToLoadProfiles, CreateYearlyProfile, \
    AggregateProfileDataForTimePeriod, TransformTimeOfUseTimeSlots, Apply
from utility import configuration
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
    TransformTypicalLoadProfile(),
    Store("typical_load_profiles_gse"),
    CalculateTypicalMonthlyConsumption(),
    Store("typical_aggregated_consumption"))).execute()

DataProcessingPipeline("users", workers=(
    ReadUserData(),
    TransformUserData(),
    Store("users"))).execute()

DataProcessingPipeline("load_profiles_from_bills", workers=(
    ReadBills(),
    TransformBills(),
    CheckAnnualSum(),
    Store("bills"),
    Write("data_users_bills"),
    TransformBillsToLoadProfiles(),
    Store("load_profiles_from_bills"),
    CreateYearlyProfile(),
    Store("yearly_load_profiles_from_bills"),
    Write("data_users_year"),
    WriteSeparately(subdirectory="Loads"))).execute()

DataProcessingPipeline("annual_consumption_to_bill_data",
                       dataset=DataStore()["users"],
                       workers=(
                           AddYearlyConsumptionToBillData(),
                           Store("users"),
                           Write("data_users"))).execute()

DataProcessingPipeline("visualize", workers=(
    Visualize("consumption_profiles", plot_consumption_profiles),)).execute()

DataProcessingPipeline("families", workers=(
    ReadBills(filename="bollette_domestici.csv"),
    TransformBills(),
    CheckAnnualSum(),
    Store("families_bills"),
    TransformBillsToLoadProfiles(),
    CreateYearlyProfile(),
    Store("yearly_load_profiles_families"),
    Write("data_families_year"),
    Apply(lambda x: x * configuration.config.getint('rec', 'number_of_families')),
    Write(f"families_{configuration.config.getint('rec', 'number_of_families')}"),
    Visualize("profiles", plot_family_profiles))).execute(user_type=UserType.PDMF)

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
    Visualize("profiles_by_month", plot_pv_profiles),
    AggregateProfileDataForTimePeriod(),
    Write("data_plants_year"),
    WriteSeparately(subdirectory="Generators"))).execute()
