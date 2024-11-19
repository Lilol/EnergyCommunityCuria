from enum import Enum

from utility.definitions import OrderedEnum


class ColumnName(OrderedEnum):
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
    CONSUMPTION = 'consumption'
    FAMILY = 'family'
    YEAR = 'year'
    SEASON = 'season'  # season (1-Winter-December to February, ...)
    MONTH = 'month'  # number of the month (1-12)
    WEEK = 'week'  # number of week of the year
    DATE = 'date' # date in pandas.DateTime format
    TIME = 'time' # time
    DAY_OF_MONTH = 'day'  # number of day in the month (1-28, 29 30, 31)
    DAY_TYPE = 'day_type'  # type of day (0-work, 1-Saturday, 2-Sunday/holiday)
    DAY_OF_WEEK = 'day_week'  # number of day in the week (1-Monday, ...)
    HOUR = 'hour'
    BILL_TYPE = 'bill_type'
    MONO_TARIFF = 'mono_tariff'
    TARIFF_TIME_SLOT = "tariff_time_slot"


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
