from utility.definitions import OrderedEnum


class DataKind(OrderedEnum):
    USER = 'user'
    USER_DATA = 'user_data'
    MUNICIPALITY = 'municipality'
    DESCRIPTION = 'description'
    USER_TYPE = 'type'  # type of end user
    USER_ADDRESS = 'address'  # address of the end user
    POWER = 'power'  # size of the plant / available power of the end-user (kW)
    ANNUAL_ENERGY = 'energy'  # annual energy produced / consumed (kWh)
    ANNUAL_YIELD = 'yield'  # specific annual production (kWh/kWp)
    TOU_ENERGY = "energy"
    PRODUCTION = 'production'
    CONSUMPTION_OF_USERS = 'consumption_of_users'
    CONSUMPTION_OF_FAMILIES = 'consumption_of_families'
    CONSUMPTION = "consumption"
    SHARED = 'shared'
    NUMBER_OF_FAMILIES = 'number_of_families'
    BATTERY_SIZE = 'battery_size'
    YEAR = 'year'
    SEASON = 'season'  # season (1-Winter-December to February, ...)
    MONTH = 'month'  # number of the month (1-12)
    WEEK = 'week'  # number of week of the year
    DATE = 'date'  # date in pandas.DateTime format
    TIME = 'time'  # time
    DAY_OF_MONTH = 'day'  # number of day in the month (1-28, 29 30, 31)
    DAY_TYPE = 'day_type'  # type of day (0-work, 1-Saturday, 2-Sunday/holiday)
    COUNT = 'count'  # Generic count type
    DAY_OF_WEEK = 'day_week'  # number of day in the week (1-Monday, ...)
    HOUR = 'hour'
    BILL_TYPE = 'bill_type'
    MONO_TARIFF = 'mono_tariff'
    TARIFF_TIME_SLOT = "tariff_time_slot"
    TOU = "tou"


class BillType(OrderedEnum):
    MONO = 'mono'
    TIME_OF_USE = 'tou'
    INVALID = 'invalid'


class UserType(OrderedEnum):
    PDMF = 'dom'
    PAUF = 'bta'
    PICM = 'ip'
    PV = 'pv'


class PvDataSource(OrderedEnum):
    PVGIS = "PVGIS"
    PVSOL = "PVSOL"
