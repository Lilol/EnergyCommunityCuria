from calendar import weekday

import holidays

from utility import configuration

weekdays = (0, 1, 2, 3, 4)
weekend = (5, 6)

year = configuration.config.getint("time", "year")


def get_weekday_code(day):
    # 0: Weekdays
    # 1: Saturday
    # 2: Sunday and holiday
    hds = holidays.country_holidays(configuration.config.get("global", "country"), years=day.year)
    if day in hds:
        return 2
    elif weekday(day.year, day.month, day.day) in weekdays:
        return 0
    else:
        return 1
