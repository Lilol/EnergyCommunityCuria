from utility import configuration

_dt = configuration.config.get("time", "resolution")

def power_to_energy(p, dt=_dt):
    """
    Calculates the total energy given power (p) and time step (dt).
    Calculate energy by summing up power values multiplied by the time interval given in hours
    """
    return p.sum() * dt
