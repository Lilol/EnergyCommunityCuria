from calendar import weekday

import holidays
from pandas import date_range, DataFrame

from utility import configuration
from input.definitions import ColumnName

weekdays = (0, 1, 2, 3, 4)
weekend = (5, 6)

year = configuration.config.getint("time", "year")


def get_weekday_code(day):
    # 0: Weekdays
    # 1: Saturday
    # 2: Sunday and holidays
    hds = holidays.country_holidays(configuration.config.get("global", "country"), years=day.year)
    if day in hds:
        return 2
    elif weekday(day.year, day.month, day.day) in weekdays:
        return 0
    else:
        return 1


def create_reference_year_dataframe():
    ref_year = configuration.config.get("time", "year")
    ref_df = DataFrame(index=date_range(start=f"{ref_year}-01-01", end=f"{ref_year}-12-31", freq="d"),
                        columns=[ColumnName.YEAR, ColumnName.MONTH, ColumnName.DAY_OF_MONTH, ColumnName.WEEK,
                                 ColumnName.SEASON])
    ref_df[ColumnName.YEAR] = ref_df.index.year
    ref_df[ColumnName.MONTH] = ref_df.index.month
    ref_df[ColumnName.DAY_OF_MONTH] = ref_df.index.day
    ref_df[ColumnName.DAY_TYPE] = ref_df.index.map(get_weekday_code)
    ref_df[ColumnName.WEEK] = ref_df.index.isocalendar().week
    ref_df[ColumnName.SEASON] = ref_df.index.month % 12 // 3 + 1
    ref_df[ColumnName.DAY_OF_WEEK] = ref_df.index.dayofweek
    return ref_df


df_year = create_reference_year_dataframe()

cols_to_add = [ColumnName.YEAR, ColumnName.MONTH, ColumnName.DAY_OF_MONTH, ColumnName.WEEK, ColumnName.SEASON,
                ColumnName.DAY_OF_WEEK]
