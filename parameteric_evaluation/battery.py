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
        for h, (p, c) in enumerate(zip(p_prod, p_cons)):
            # Power to charge the BESS (discharge if negative)
            b = p - c
            # Correct according to technical/physical limits
            if b < 0:
                b = max(b, -e_stored, -p_bess_max)
            else:
                b = min(b, self._size - e_stored, p_bess_max)
            # Update BESS power array and stored energy
            p_bess[h] = b
            e_stored = e_stored + b

        return p_bess

    def get_as_optim(self):
        pass
