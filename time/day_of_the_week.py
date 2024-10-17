from calendar import weekday

import holidays

weekday_name = {0: "Mon", 1: "Tue", 2:"Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
weekday_long_name = {0: "Monday", 1: "Tuesday", 2: "Wednesday",  3:"Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
weekdays = (0, 1, 2, 3, 4)
weekend = (5, 6)


def get_weekday_code(day):
    hds = holidays.country_holidays("Italy", years=day.year)
    return weekday(day.year, day.month, day.day) if day not in hds else 6


def get_weekday_codes():
    return weekdays


def get_weekend_codes():
    return weekend


def get_similar_day_code(wdc):
    if wdc in weekdays:
        return weekdays
    else:
        return weekend
