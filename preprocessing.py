from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from data_storage.data_store import DataStore
from data_storage.store_data import Store
from io_operation.input import UserType
from io_operation.input import ReadUserData, ReadBills, ReadPvPlantData, ReadTariff, ReadTypicalLoadProfile, ReadProduction
from io_operation.output import Write, WriteSeparately
from transform.check import CheckAnnualSum
from transform.combine.combine import CalculateTypicalMonthlyConsumption, AddYearlyConsumptionToBillData
from transform.extract.data_extractor import ExtractTimeOfUseParameters, ExtractDayTypesInTimeframe, \
    ExtractDayCountInTimeframe, ExtractTimeOfUseTimeSlotCountByDayType, \
    ExtractTimeOfUseTimeSlotCountByMonth
from transform.transform import TransformTariffData, TransformTypicalLoadProfile, TransformUserData, \
    TransformPvPlantData, TransformBills, TransformProduction, TransformBillsToLoadProfiles, CreateYearlyProfile, \
    AggregateProfileDataForTimePeriod, Apply
from utility import configuration
from utility.init_logger import init_logger
from visualization.preprocessing_visualization import plot_family_profiles, plot_pv_profiles, plot_consumption_profiles
from visualization.visualize import Visualize

init_logger()

# ----------------------------------------------------------------------------
DataProcessingPipeline("day_properties", workers=(
    ExtractDayTypesInTimeframe(),
    Store("day_types"),
    ExtractDayCountInTimeframe(),
    Store("day_count"))).execute()

DataProcessingPipeline("time_of_use", workers=(
    ReadTariff(),
    TransformTariffData(),
    ExtractTimeOfUseParameters(),
    Store("time_of_use_time_slots"),
    ExtractTimeOfUseTimeSlotCountByDayType(),
    Store("time_of_use_time_slot_count_by_day_type"),
    ExtractTimeOfUseTimeSlotCountByMonth(),
    Store("time_of_use_time_slot_count_by_month"))).execute()

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
    WriteSeparately(subdirectory="Loads"),
    AggregateProfileDataForTimePeriod(),
    Write("data_users_tou"))).execute()

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
    Visualize("profiles", plot_family_profiles),
    AggregateProfileDataForTimePeriod(),
    Write("data_families_tou"))).execute(user_type=UserType.PDMF)

n_families = configuration.config.getint('rec', 'number_of_families')
DataProcessingPipeline(f"yearly_load_profiles_families_{n_families}",
                       dataset=DataStore()["yearly_load_profiles_families"],
                       workers=(
                           Apply(operation=lambda x: x * n_families),
                           Write(f"families_{n_families}"))).execute()

DataProcessingPipeline("pv_plants", workers=(
    ReadPvPlantData(),
    TransformPvPlantData(),
    Store("pv_plants"),
    Write("data_plants"))).execute()

DataProcessingPipeline("pv_production", workers=(
    ReadProduction(),
    TransformProduction(),
    # ExtractTypicalYear(),
    # Store("pv_profiles"),
    # CreateYearlyProfile(),
    # Write("data_plants_year"),
    Visualize("profiles_by_month", plot_pv_profiles),
    WriteSeparately(subdirectory="Generators"),
    AggregateProfileDataForTimePeriod(),
    Write("data_plants_tou"))).execute()
