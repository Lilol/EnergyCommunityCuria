import logging
from typing import Iterable

import cvxopt as opt
import numpy as np
import numpy.linalg as lin
import xarray as xr

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from operation.definitions import Status, ScalingMethod
from operation.operation import Operation
from utility import configuration

logger = logging.getLogger(__name__)


class ScaleProfile(Operation):
    _name = "profile_scaler"
    _method = ScalingMethod.INVALID
    subclasses = {}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._method] = cls

    def __new__(cls, *args, **kwargs):
        return cls.subclasses[configuration.config.get("profile", "scaling_method")](*args, **kwargs)


class ScaleInProportion(ScaleProfile):
    _name = "flat_tariff_profile_scaler"
    _method = ScalingMethod.IN_PROPORTION

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        self.status = Status.OPTIMAL
        reference_profile = operands[0]
        total_consumption_by_time_slots = operands[1]
        return reference_profile.values / reference_profile.sum() * total_consumption_by_time_slots.sum()


class ScaleFlat(ScaleProfile):
    _name = "flat_scaler"
    _method = ScalingMethod.FLAT

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

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
        time_of_use_time_slots = DataStore()["time_of_use_time_slots"]
        scaled_profile = time_of_use_time_slots.copy()
        for tariff_time_slot in configuration.config.getarray("tariff", "tariff_time_slots", int):
            scaled_profile.where(time_of_use_time_slots != tariff_time_slot).fillna(
                total_consumption_by_time_slots[tariff_time_slot] / np.count_nonzero(
                    time_of_use_time_slots == tariff_time_slot))
        return scaled_profile


class ScaleTimeOfUseProfile(ScaleProfile):
    _name = "time_of_use_tariff_profile_scaler"
    _method = ScalingMethod.TIME_OF_USE

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

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
        total_reference_consumption_by_time_slots = operands[2]

        # calculate scaling factors (one for each tariff time-slot)
        scaling_factor = total_consumption_by_time_slots.squeeze().values / total_reference_consumption_by_time_slots
        scaling_factor[scaling_factor.isnull()] = 0

        # evaluate load profiles by scaling the reference profiles
        time_of_use_time_slots = DataStore()["time_of_use_time_slots"]
        scaled_profile = xr.concat(
            [reference_profile.where(time_of_use_time_slots == time_slot) * scaling_factor[time_slot] for time_slot in
             configuration.config.getarray("tariff", "tariff_time_slots", int)], dim="tariff_time_slot").sum(
            "tariff_time_slot", skipna=True)

        # Substituting missing values with a flat consumption
        for s in scaling_factor[scaling_factor == 0]:
            scaled_profile.where(time_of_use_time_slots != s.tariff_time_slot).fillna(
                total_consumption_by_time_slots[s.tariff_time_slot] / np.count_nonzero(
                    time_of_use_time_slots == s.tariff_time_slot))

        self.status = Status.OPTIMAL
        return scaled_profile


class ScaleByLinearEquation(ScaleProfile):
    _name = "linear_scaler"
    _method = ScalingMethod.LINEAR_EQUATION

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
        e_ref = DataStore()["typical_aggregated_consumption"]

        # scaling factors to respect total consumption
        self.status = Status.OPTIMAL
        try:
            scaling_factor = np.dot(lin.inv(e_ref.values), total_consumption_by_time_slots.values[:, np.newaxis])
            if np.any(scaling_factor < 0):
                self.status = Status.UNPHYSICAL
        except lin.LinAlgError as e:
            scaling_factor = -1
            self.status = Status.ERROR
            logger.warning(f"Error during optimization '{e}'")
        # ------------------------------------
        # get load profiles in day-types and return
        return reference_profile * scaling_factor.flatten()[:, np.newaxis]


class ScaleByQuadraticOptimization(ScaleProfile):
    _name = "quadratic_optimization_scaler"
    _method = ScalingMethod.QUADRATIC_OPTIMIZATION

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        """
        # This method evaluates the load profile using a simple quadratic optimization
        # problem, where the sum of the deviations over subsequent time steps is to be
        # minimised. The total consumption is set as a constraint.
        # Moreover, other constraints can be added.
        ____________
        DESCRIPTION
        The function evaluates hourly load profiles (y) in each type of day (j)
        scaling given reference load profiles (y_ref) in order to respect the
        monthly energy consumption divided into tariff time-slots (x).
        ______
        NOTES
        The method solves a quadratic optimisation problem where the deviation
        from the given reference profiles is to be minimised.
        The total consumption is a constraint. Other (optional) constraints are:
            - demand smaller than a maximum value in all time-steps;
        ____________
        PARAMETERS
        x : np.ndarray
            Monthly electricity consumption divided into tariff time-slots
            Array of shape (nf,) where 'nf' is the number of tariff time-slots.
        nd : np.ndarray
            Number of days of each day-type in the month
            Array of shape (nj,) where 'nj' is the number of day-types
            (according to ARERA's subdivision into day-types).
        y_ref : np.ndarray
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day, containing the reference profiles.
        y_max : float or None, optional
            Maximum value that the demand can assume.
            If None, the related constraint is not applied. Default is None.
            NOTE : convergence may not be reached if y_max is too small.
        obj : int, optional
            Objective of the optimisation
            0 : minimise sum(y[i] - y_ref[i])**2;
            1 : minimise sum(y[i]/y_ref[i] -1)**2.
            Default is 0.
        obj_reg : float or None, optional
            Weight of "regularisation" term to the objective
            Regularisation is intended as difference in the demand between
            subsequent time-steps, i.e.: obj_reg * sum(y[i] - y[(i+1)%nh])**2.
            Default is None, same as 0.
        cvxopt : dict, optional
            Optional parameters to pass to 'cvxopt'.
        _______
        RETURNS
        y : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day.
        status : str
            Status of the optimisation
            Can be : 'optimal', 'unknown', 'unfeasible'.
        _____
        INFO
        Author : G. Lorenti (gianmarco.lorenti@polito.it)
        Date : 17.11.2022 (last update: 17.11.2022)
        """
        # TODO: implement it in xarray
        total_reference_consumption_by_time_slots = operands[0]
        reference_profile = operands[1]
        total_consumption_by_time_slots = operands[2]
        y_max = kwargs.pop("y_max", None)
        obj = kwargs.pop("obj", None)
        obj_reg = kwargs.pop("obj_reg", None)
        cvxopt = kwargs.pop("cvxopt", {})
        # ------------------------------------
        # total consumption in the reference profile in the month
        # x_ref = np.array([
        #     sum([np.sum(y_ref.reshape(nj, ni)[ij, arera[ij]==f])*nd[ij]
        #           for ij in range(nj)]) for f in fs])
        # y_ref = y_ref * np.sum(x) / np.sum(x_ref)
        # ------------------------------------
        # constraint in the optimsation problem
        # NOTE : 'nh' variables are to be found
        # coefficients and known term of equality constraints
        # NOTE : equality constraints are written as <a,x> = b
        n_time_steps = configuration.config.getint("time", "number_of_time_steps_per_day")
        a = np.zeros((0, n_time_steps))
        b = np.zeros((0,))
        # coefficients and known term of inequality constraint
        # NOTE : inequality constraints are written as <g,x> <= h
        g = np.zeros((0, n_time_steps))
        h = np.zeros((0,))
        number_of_time_of_use_periods = configuration.config.getint("time", "number_of_time_of_use_periods")
        time_of_use_time_slots = DataStore()["time_of_use_time_slots"]
        # constraint on the total consumption (one for each tariff time slot)
        for if_, f in enumerate(number_of_time_of_use_periods):
            aux = np.concatenate([(time_of_use_time_slots[ij] == f) * nd[ij] for ij in range(nj)])
            a = np.concatenate((a, aux[np.newaxis, :]), axis=0)
            b = np.append(b, total_consumption_by_time_slots[if_])
        # constraint for variables to be positive
        g = np.concatenate((g, -1 * np.eye(n_time_steps)))
        h = np.concatenate((h, np.zeros((n_time_steps,))))
        # constraint on maximum power
        if y_max:
            assert isinstance(y_max, (float, int)) and y_max > 0, "If given, 'y_max' must be a positive number."
            g = np.concatenate((g, np.eye(n_time_steps)))
            h = np.concatenate((h, y_max * np.ones((n_time_steps,))))
        # ------------------------------------
        # objective function
        # NOTE : objective is written as min 0.5*<<x.T, p>, x> + <q.T, x>
        assert obj in (objs := list(range(2))), f"If given, 'obj' must be in: {', '.join([str(it) for it in objs])}."
        # if obj == 0:
        # p = np.eye(nh)
        p = np.diag(np.repeat(nd / nd.sum(), ni))
        q = -reference_profile * np.repeat(nd / nd.sum(), ni)
        # q = -y_ref
        # else:
        #     delta = 1e-1
        #     y_ref = y_ref + delta
        #     p = np.diag(1/y_ref**2)
        #     q = -1/y_ref
        # regularisation term in the objective
        if obj_reg:
            assert isinstance(obj_reg, float), "If given 'obj_reg' must be a float (weight)."
            l = np.zeros((n_time_steps, n_time_steps))
            for i_h in range(n_time_steps):
                l[i_h, i_h] = 1
                l[i_h, (i_h + 1) % n_time_steps] = -1
            p += obj_reg * np.dot(l.T, l)  # q += np.zeros((nh,))
        # options for solver
        cvxopt = {"kktsolver": 'ldl', "options": {'kktreg': 1e-9, "show_progress": False}, **cvxopt}
        # solve and retrieve solution
        sol = opt.solvers.qp(P=opt.matrix(p), q=opt.matrix(q), A=opt.matrix(a), b=opt.matrix(b), G=opt.matrix(g),
                             h=opt.matrix(h), **cvxopt)
        self.status = Status(sol['status'])
        return OmnesDataArray(np.array(sol['x']).flatten())
