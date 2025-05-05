import logging
from typing import Iterable

import numpy as np
from numpy import linalg as lin

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from operation import ScaleProfile
from operation.definitions import ScalingMethod, Status

logger = logging.getLogger(__name__)

class ScaleByLinearEquation(ScaleProfile):
    _name = "linear_scaler"
    _key = ScalingMethod.LINEAR_EQUATION

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        """
        Function 'scale_seteq'
        ____________
        DESCRIPTION
        The function evaluates hourly load profiles (y_scale) in each type of
        day (j) scaling given reference load profiles (y_ref) in order to respect
        the monthly energy consumption divided into tariff time-slots (x).
        ______
        NOTES
        The method solves a set of linear equations to find one scaling factor (k)
        for each day-type, which are multiplied by the reference load profiles
        assigned to each day-type to respect the total consumption.
        NOTE : the problem might not have a solution or have un-physical solution
        (e.g. negative scaling factors).
        ____________
        PARAMETERS
        x : np.ndarray
            Monthly aggregated electricity consumption divided into tariff time-slots
            Array of shape (nf,) where 'nf' is the number of tariff time-slots.
        _______
        RETURNS
        y_scal : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each day.
        status : str
            Status of the solution.
            Can be : 'ok', 'unphysical', 'error'.
        """
        # evaluate scaling factors to get load profiles
        total_consumption_by_time_slots = operands[0]
        reference_profile = operands[1]
        # energy consumed in reference profiles assigned to each typical day,
        # divided into tariff time-slots
        aggregated_consumption_of_reference_profile = operands[2]

        # scaling factors to respect total consumption
        self.status = Status.OPTIMAL
        try:
            scaling_factor = OmnesDataArray(np.dot(lin.inv(aggregated_consumption_of_reference_profile.T.values),
                                                   total_consumption_by_time_slots.squeeze().values),
                                            dims=DataKind.DAY_TYPE.value, coords={
                    DataKind.DAY_TYPE.value: reference_profile[DataKind.DAY_TYPE.value].values})
            if np.any(scaling_factor < 0):
                self.status = Status.UNPHYSICAL
        except lin.LinAlgError as e:
            scaling_factor = OmnesDataArray(-1., dims=(DataKind.DAY_TYPE.value,), coords={
                DataKind.DAY_TYPE.value: reference_profile[DataKind.DAY_TYPE.value].values})
            self.status = Status.ERROR
            logger.warning(f"Error during calculating scaling factor '{e}'")
        # ------------------------------------
        # get load profiles in day-types and return
        return reference_profile * scaling_factor
