from pandas import Period

from utility import configuration


def convert_to_hours(period_str):
    period = Period(freq=period_str)
    return period.hour


_dt = convert_to_hours(configuration.config.get("time", "resolution"))


def power_to_energy(p, dt=_dt):
    """
    Calculates the total energy given power (p) and time step (dt).
    Calculate energy by summing up power values multiplied by the time interval given in hours
    """
    return p * dt
