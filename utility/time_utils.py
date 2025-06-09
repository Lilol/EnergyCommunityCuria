from pandas import Timestamp, to_timedelta, Timedelta

from utility.configuration import config


def days_of_year(year):
    return Timestamp(year, 12, 31).dayofyear


def n_periods_in_interval(interval_length: str, period_length: str) -> int:
    interval = conventionalize_time_period(interval_length)
    step = conventionalize_time_period(period_length)
    return int(interval / step)


def conventionalize_time_period(interval_length):
    if 'y' in interval_length.lower():
        year_in_days = days_of_year(config.getint("time", "year"))
        interval = to_timedelta(year_in_days, unit='D')
    else:
        interval = Timedelta(interval_length)
    return interval
