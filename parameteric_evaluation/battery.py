from typing import Iterable

import numpy as np
import xarray as xr

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import BatteryPowerFlows, OtherParameters


class Battery(Calculator):
    def __init__(self, size, t_min=None):
        self._size = size
        self.p_max = np.inf if t_min is None else self._size / t_min

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        return Battery(kwargs.pop('size'), t_min=kwargs.pop('t_min')).manage_bess(input_da)

    def manage_bess(self, dataset):
        """Manage BESS power flows to increase shared energy."""
        # Initialize BESS power and dataarray of power flows of all users in the system
        bess_power = OmnesDataArray(0, dims=dataset.dims, coords={**dataset.coords,
                                                                  DataKind.CALCULATED.value: [bess_metr for bess_metr in
                                                                                              BatteryPowerFlows]})
        dataset = xr.concat([dataset, bess_power], dim=DataKind.CALCULATED.value)

        if self._size == 0:
            return dataset

        # Manage flows in all time steps
        for array in dataset.transpose(DataKind.TIME.value, ...):
            # Power to charge the BESS (discharge if negative)
            charging_power = array.sel({DataKind.CALCULATED.value: OtherParameters.INJECTED_ENERGY}) - array.sel(
                {DataKind.CALCULATED.value: OtherParameters.WITHDRAWN_ENERGY})
            # Correct according to technical/physical limits
            e_stored = array.sel({DataKind.CALCULATED.value: BatteryPowerFlows.STORED_ENERGY})
            if charging_power < 0:
                charging_power = max(charging_power, -e_stored, -self.p_max)
            else:
                charging_power = min(charging_power, self._size - e_stored, self.p_max)

            # Update BESS power array and stored energy
            dataset = dataset.update((charging_power, e_stored + charging_power, array.sel(
                {DataKind.CALCULATED.value: OtherParameters.INJECTED_ENERGY}) - charging_power), **{
                DataKind.CALCULATED.value: [BatteryPowerFlows.POWER_CHARGE, BatteryPowerFlows.STORED_ENERGY,
                                            OtherParameters.INJECTED_ENERGY], DataKind.TIME.value: array.time})

        return dataset
