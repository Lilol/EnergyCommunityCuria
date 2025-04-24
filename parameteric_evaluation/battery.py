import numpy as np
import xarray as xr

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation.definitions import BatteryPowerFlows, PhysicalMetric


class Battery:
    def __init__(self, size):
        self._size = size

    def manage_bess(self, dataset, t_min=None):
        """Manage BESS power flows to increase shared energy."""
        # Initialize BESS power array and stored energy
        bess_power = OmnesDataArray(0, dims=dataset.dims, coords={**dataset.coords,
                                                                  DataKind.CALCULATED.value: (bess_metr for bess_metr in
                                                                                              BatteryPowerFlows), })
        dataset = xr.concat([dataset, bess_power], dim=DataKind.CALCULATED.value)

        if self._size == 0:
            return dataset

        # Get inputs
        p_bess_max = np.inf if t_min is None else self._size / t_min

        # Manage flows in all time steps
        for array in dataset.transpose(DataKind.TIME.value, ...):
            # Power to charge the BESS (discharge if negative)
            charging_power = array.sel({DataKind.CALCULATED.value: PhysicalMetric.INJECTED_ENERGY}) - array.sel(
                {DataKind.CALCULATED.value: PhysicalMetric.WITHDRAWN_ENERGY})
            # Correct according to technical/physical limits
            e_stored = array.sel({DataKind.CALCULATED.value: BatteryPowerFlows.STORED_ENERGY})
            if charging_power < 0:
                charging_power = max(charging_power, -e_stored, -p_bess_max)
            else:
                charging_power = min(charging_power, self._size - e_stored, p_bess_max)

            # Update BESS power array and stored energy
            dataset[array.time.item(), :, BatteryPowerFlows.POWER_CHARGE] = charging_power
            dataset[array.time.item(), :, BatteryPowerFlows.STORED_ENERGY] = e_stored + charging_power

        return dataset

    def get_as_optim(self):
        pass
