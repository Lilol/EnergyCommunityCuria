from enum import Enum


class InputColumn(Enum):
    USER = 'user'
    MUNICIPALITY = 'municipality'
    DESCRIPTION = 'description'
    USER_TYPE = 'type'  # type of end user
    USER_ADDRESS = 'address'  # address of the end user
    POWER = 'power'  # size of the plant / available power of the end-user (kW)
    ANNUAL_ENERGY = 'energy'  # annual energy produced / consumed (kWh)
    ANNUAL_YIELD = 'yield'  # specific annual production (kWh/kWp)
    TOU_ENERGY = "energy"
    YEAR = 'year'
    SEASON = 'season'  # season (1-Winter-December to February, ...)
    MONTH = 'month'  # number of the month (1-12)
    WEEK = 'week'  # number of week of the year
    DAY_OF_MONTH = 'day'  # number of day in the month (1-28, 29 30, 31)
    DAY_TYPE = 'day_type'  # type of day (0-work, 1-Saturday, 2-Sunday/holiday)
    DAY_OF_WEEK = 'day_week'  # number of day in the week (1-Monday, ...)


class BillType(Enum):
    MONO = 'mono'
    TIME_OF_USE = 'tou'
    INVALID = 'invalid'


class UserType(Enum):
    BTA = 'bta'
    DOMESTIC = 'dom'
    IP = 'ip'


class PvDataSource(Enum):
    PVGIS = "PVGIS"
    PVSOL = "PVSOL"
