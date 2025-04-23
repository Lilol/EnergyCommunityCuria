import numpy as np


class Battery:
    def __init__(self, size):
        self._size = size

    def manage_bess(self, p_prod, p_cons, t_min=None):
        """Manage BESS power flows to increase shared energy."""
        # Initialize BESS power array and stored energy
        p_bess = np.zeros_like(p_prod)
        if self._size == 0:
            return p_bess

        # Get inputs
        p_bess_max = np.inf if t_min is None else self._size / t_min
        e_stored = 0

        # Manage flows in all time steps
        for h, (prod, cons) in enumerate(zip(p_prod, p_cons)):
            # Power to charge the BESS (discharge if negative)
            charging_power = prod - cons
            # Correct according to technical/physical limits
            if charging_power < 0:
                charging_power = max(charging_power, -e_stored, -p_bess_max)
            else:
                charging_power = min(charging_power, self._size - e_stored, p_bess_max)
            # Update BESS power array and stored energy
            p_bess[h] = charging_power
            e_stored = e_stored + charging_power

        return p_bess

    def get_as_optim(self):
        pass
