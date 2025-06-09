import logging

import numpy as np
import xarray as xr

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import BatteryPowerFlows, OtherParameters

logger = logging.getLogger(__name__)


class Battery(Calculator):
    def __init__(self, size, t_min=None):
        self._size = size
        self.p_max = np.inf if t_min is None else self._size / t_min

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        return Battery(kwargs.pop('size'), t_min=kwargs.pop('t_min')).manage_bess(
            input_da), results_of_previous_calculations

    def manage_bess(self, dataset: OmnesDataArray) -> OmnesDataArray:
        calc_dim = DataKind.CALCULATED.value
        time_dim = DataKind.TIME.value

        bess_coords = {**dataset.coords, calc_dim: [m for m in BatteryPowerFlows if m.value != "invalid"]}
        bess_power = OmnesDataArray(
            0.0,
            dims=dataset.dims,
            coords=bess_coords
        )
        dataset = xr.concat([dataset, bess_power], dim=calc_dim)

        if self._size == 0:
            return dataset

        # Get slices
        inj = dataset.sel({calc_dim: OtherParameters.INJECTED_ENERGY})
        withdrawn = dataset.sel({calc_dim: OtherParameters.WITHDRAWN_ENERGY})
        stored = np.zeros_like(inj)

        charge = inj - withdrawn
        new_inj = inj.copy()
        bess_charge = np.zeros_like(inj)

        for t in range(len(inj[time_dim])):
            power = charge.isel({time_dim: t})
            e = stored[t - 1] if t > 0 else 0

            # Apply BESS constraints
            limited = xr.where(
                power < 0,
                xr.ufuncs.maximum(power, xr.ufuncs.maximum(-e, -self.p_max)),
                xr.ufuncs.minimum(power, xr.ufuncs.minimum(self._size - e, self.p_max))
            )
            bess_charge[t] = limited
            stored[t] = e + limited
            new_inj[t] = inj.isel({time_dim: t}) - limited

        dataset.loc[{calc_dim: BatteryPowerFlows.POWER_CHARGE}] = bess_charge
        dataset.loc[{calc_dim: BatteryPowerFlows.STORED_ENERGY}] = stored
        dataset.loc[{calc_dim: OtherParameters.INJECTED_ENERGY}] = new_inj

        return dataset

