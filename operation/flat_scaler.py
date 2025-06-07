from typing import Iterable

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from operation import ScaleProfile
from operation.definitions import ScalingMethod, Status
from utility import configuration


class ScaleFlat(ScaleProfile):
    _name = "flat_tariff_profile_scaler"
    _key = ScalingMethod.FLAT

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.status = Status.OPTIMAL

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        """
        ____________
        DESCRIPTION
        The function evaluates hourly load profiles (y) in each type of day (j)
        from the monthly energy consumption divided into tariff time-slots (x).
        ______
        NOTES
        The method assumes the same demand within all time-steps in the same
        tariff time-slot (f), hence it just spreads the total energy consumption
        according to the number of hours of each tariff time-slot in the month.
        ___________
        PARAMETERS
        total_consumption_by_tariff_slots : np.ndarray
            Monthly electricity consumption divided into tariff time-slots
            Array of shape (nf, ) where 'nf' is the number of tariff time-slots.
        number_of_days_by_type : np.ndarray
            Number of days of each day-type in the month
            Array of shape (nj, ) where 'nj' is the number of day-types (according to ARERA's subdivision into day-types).
        ________
        RETURNS
        y : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day.
        _____
        INFO
        Author : G. Lorenti (gianmarco.lorenti@polito.it)
        Date : 16.11.2022
        """
        # ------------------------------------
        total_consumption_by_time_slots = operands[0]
        reference_profile = operands[1]
        data_store = DataStore()
        time_of_use_time_slots = data_store["time_of_use_time_slots"]
        time_of_use_time_slot_count_by_month = data_store["time_of_use_time_slot_count_by_month"].sel(
            {DataKind.MONTH.value: reference_profile.month.values})

        scaled_profile = time_of_use_time_slots.copy()
        for tariff_time_slot, tou_label in zip(configuration.config.getarray("tariff", "tariff_time_slots", int),
                                               configuration.config.get("tariff", "time_of_use_labels")):
            scaled_profile = scaled_profile.where(time_of_use_time_slots != tariff_time_slot).fillna(
                total_consumption_by_time_slots.loc[:, :, tou_label].squeeze() / (
                    time_of_use_time_slot_count_by_month.sel({DataKind.TARIFF_TIME_SLOT.value: tariff_time_slot})))
        return scaled_profile
