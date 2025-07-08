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
        bess_power = OmnesDataArray(0.0, dims=dataset.dims, coords=bess_coords)
        dataset = xr.concat([dataset, bess_power], dim=calc_dim)

        if self._size == 0:
            return dataset

        inj = dataset.sel({calc_dim: OtherParameters.INJECTED_ENERGY})
        withdrawn = dataset.sel({calc_dim: OtherParameters.WITHDRAWN_ENERGY})

        charge = inj - withdrawn
        time_vals = inj[time_dim].values

        new_inj_list = []
        bess_charge_list = []
        stored_list = []

        e = xr.zeros_like(charge.isel({time_dim: 0}))

        for t, time_val in enumerate(time_vals):
            power = charge.sel({time_dim: time_val})

            charge_max = xr.where(
                power < 0,
                xr.ufuncs.maximum(power, xr.ufuncs.maximum(-e, -self.p_max)),
                xr.ufuncs.minimum(power, xr.ufuncs.minimum(self._size - e, self.p_max))
            )

            new_e = e + charge_max
            new_inj_t = inj.sel({time_dim: time_val}) - charge_max

            # Assign time coordinate to each slice explicitly
            for arr, name in [(charge_max, "charge"), (new_e, "stored"), (new_inj_t, "inj")]:
                arr.coords[time_dim] = time_val  # Assign scalar coordinate

            # Append
            bess_charge_list.append(charge_max.expand_dims(time_dim))
            stored_list.append(new_e.expand_dims(time_dim))
            new_inj_list.append(new_inj_t.expand_dims(time_dim))

            e = new_e  # update energy for next step

        # Concatenate over time
        bess_charge = xr.concat(bess_charge_list, dim=time_dim)
        stored = xr.concat(stored_list, dim=time_dim)
        new_inj = xr.concat(new_inj_list, dim=time_dim)

        dataset.loc[{calc_dim: BatteryPowerFlows.POWER_CHARGE}] = bess_charge
        dataset.loc[{calc_dim: BatteryPowerFlows.STORED_ENERGY}] = stored
        dataset.loc[{calc_dim: OtherParameters.INJECTED_ENERGY}] = new_inj

        return dataset
