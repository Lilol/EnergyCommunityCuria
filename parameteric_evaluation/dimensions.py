from utility import configuration


def convert_to_hours(time_str):
    time_str = time_str.strip().lower()
    if time_str.endswith('h'):
        return float(time_str[:-1])
    elif time_str.endswith('min'):
        return float(time_str[:-3]) / 60
    elif time_str.endswith('sec') or time_str.endswith('s'):
        if time_str.endswith('sec'):
            return float(time_str[:-3]) / 3600
        else:
            return float(time_str[:-1]) / 3600
    else:
        raise ValueError(f"Unsupported time format: {time_str}")


_dt = convert_to_hours(configuration.config.get("time", "resolution"))


def power_to_energy(p, dt=_dt):
    """
    Calculates the total energy given power (p) and time step (dt).
    Calculate energy by summing up power values multiplied by the time interval given in hours
    """
    return p * dt
