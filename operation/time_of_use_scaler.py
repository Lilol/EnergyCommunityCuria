from typing import Iterable

import numpy as np
import xarray as xr

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from operation import ScaleProfile
from operation.definitions import ScalingMethod, Status
from utility import configuration


class ScaleTimeOfUseProfile(ScaleProfile):
    _name = "time_of_use_tariff_profile_scaler"
    _key = ScalingMethod.TIME_OF_USE

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.status = Status.OPTIMAL

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        """
        ____________
        DESCRIPTION
        The function evaluates hourly load profiles (y_scale) in each type of
        day (j) scaling given reference load profiles (y_ref) in order to respect
        the monthly energy consumption divided into tariff time-slots (x).
        ______
        NOTES
        The method evaluates one scaling factor for each tariff time-slot, which
        is equal to the monthly consumption of y_ref in that tariff time-slot
        divided by the consumption in that tariff time slot associated with the reference load profile.
        The latter is then scaled separately for the time-steps in each time-slot.
        ____________
        PARAMETERS
        operands : Iterable[OmnesDataArray]
            Monthly electricity consumption divided into tariff time-slots
            Array of shape (nf,) where 'nf' is the number of tariff time-slots.
        y_ref : np.ndarray
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day, containing the reference profiles.
        _______
        RETURNS
        y_scale : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day.
        """
        # ------------------------------------
        # scale reference profiles
        # evaluate the monthly consumption associated with the reference profile
        # divided into tariff time-slots
        total_consumption_by_time_slots = operands[0]
        reference_profile = operands[1]
        total_reference_consumption_by_time_slots = operands[2].sum(DataKind.DAY_TYPE.value)

        # calculate scaling factors (one for each tariff time-slot)
        scaling_factor = total_consumption_by_time_slots.squeeze().values / total_reference_consumption_by_time_slots
        scaling_factor[scaling_factor.isnull()] = 0

        # evaluate load profiles by scaling the reference profiles
        time_of_use_time_slots = DataStore()["time_of_use_time_slots"]
        scaled_profile = xr.concat(
            [reference_profile.where(time_of_use_time_slots == time_slot) * scaling_factor[time_slot] for time_slot in
             configuration.config.getarray("tariff", "tariff_time_slots", int)],
            dim=DataKind.TARIFF_TIME_SLOT.value).sum(DataKind.TARIFF_TIME_SLOT.value, skipna=True)

        # Substituting missing values with a flat consumption
        for s in scaling_factor[scaling_factor == 0]:
            scaled_profile.where(time_of_use_time_slots != s.tariff_time_slot).fillna(
                total_consumption_by_time_slots[s.tariff_time_slot] / np.count_nonzero(
                    time_of_use_time_slots == s.tariff_time_slot))

        return scaled_profile
