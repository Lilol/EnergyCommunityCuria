from typing import Iterable

import cvxopt as opt
import numpy as np
import numpy.linalg as lin

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from operation.definitions import Status
from operation.operation import Operation
from utility import configuration


class ProfileScaler(Operation):
    _name = "profile_scaler"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        raise NotImplementedError


class ScaleFlatTariffProfile(ProfileScaler):
    _name = "flat_tariff_profile_scaler"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        self.status = Status.OPTIMAL
        y_ref = operands[0]
        x = operands[1]
        return y_ref.values / y_ref.sum() * x.sum()


class ScaleTimeOfUseProfile(ProfileScaler):
    _name = "time_of_use_tariff_profile_scaler"

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
        # x_ref = ProfileExtractor.get_monthly_consumption(y_ref)
        x_ref = operands[0]
        y_ref = operands[1]
        x = operands[2]

        # calculate scaling factors (one for each tariff time-slot)
        scaling_factor = x / x_ref
        scaling_factor[np.isnan(scaling_factor)] = 0

        # evaluate load profiles by scaling the reference profiles
        y_scal = y_ref.copy()
        time_of_use_time_slots = DataStore()["time_of_use_time_slots"]
        # time-steps belonging to each tariff time-slot are scaled separately
        for day_type in configuration.config.getarray("time", "day_types", int):
            y_scal[time_of_use_time_slots == day_type] = y_ref[time_of_use_time_slots == day_type] * scaling_factor[
                day_type]

        if np.any(scaling_factor == 0):
            for i in np.where(scaling_factor == 0):
                y_scal[time_of_use_time_slots == i] = x[i] / np.count_nonzero(time_of_use_time_slots == i)
        # ---------------------------------------
        self.status = Status.OPTIMAL
        return y_scal


class LinearScaler(ProfileScaler):
    _name = "linear_scaler"

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
            Monthly electricity consumption divided into tariff time-slots
            Array of shape (nf,) where 'nf' is the number of tariff time-slots.
        nd : np.ndarray
            Number of days of each day-type in the month
            Array of shape (nj,) where 'nj' is the number of day-types
            (according to ARERA's subdivision into day-types).
        y_ref : np.ndarray
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day, containing the reference profiles.
        _______
        RETURNS
        y_scal : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day.
        status : str
            Status of the solution.
            Can be : 'ok', 'unphysical', 'error'.
        """
        # evaluate scaling factors to get load profiles
        y_ref = operands[0]
        # energy consumed in reference profiles assigned to each typical day,
        # divided in tariff time-slots
        e_ref = np.concatenate(
            [np.array([[y_ref[ij, arera[ij] == f].sum() * nd[ij] for ij in range(nj)]]) for _, f in enumerate(fs)],
            axis=0)
        # scaling factors to respect total consumption
        self.status = Status.OPTIMAL
        try:
            k = np.dot(lin.inv(e_ref), x[:, np.newaxis])
            if np.any(k < 0):
                self.status = Status.UNPHYSICAL
        except lin.LinAlgError:
            k = -1
            self.status = Status.ERROR
        # ------------------------------------
        # get load profiles in day-types and return
        y_scal = y_ref * k.flatten()[:, np.newaxis]
        return y_scal.flatten()


class QuadraticOptimizationScaler(ProfileScaler):
    _name = "quadratic_optimization_scaler"

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
            Maximum value that the demand can assume
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
        x = operands[0]
        y_ref = operands[2]
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
        a = np.zeros((0, nh))
        b = np.zeros((0,))
        # coefficients and known term of inequality constraint
        # NOTE : inequality constraints are written as <g,x> <= h
        g = np.zeros((0, nh))
        h = np.zeros((0,))
        # constraint on the total consumption (one for each tariff time slot)
        for if_, f in enumerate(fs):
            aux = np.concatenate([(arera[ij] == f) * nd[ij] for ij in range(nj)])
            a = np.concatenate((a, aux[np.newaxis, :]), axis=0)
            b = np.append(b, x[if_])
        # constraint for variables to be positive
        g = np.concatenate((g, -1 * np.eye(nh)))
        h = np.concatenate((h, np.zeros((nh,))))
        # constraint on maximum power
        if y_max:
            assert isinstance(y_max, (float, int)) and y_max > 0, "If given, 'y_max' must be a positive number."
            g = np.concatenate((g, np.eye(nh)))
            h = np.concatenate((h, y_max * np.ones((nh,))))
        # ------------------------------------
        # objective function
        # NOTE : objective is written as min 0.5*<<x.T, p>, x> + <q.T, x>
        assert obj in (objs := list(range(2))), f"If given, 'obj' must be in: {', '.join([str(it) for it in objs])}."
        # if obj == 0:
        # p = np.eye(nh)
        p = np.diag(np.repeat(nd / nd.sum(), ni))
        q = -y_ref * np.repeat(nd / nd.sum(), ni)
        # q = -y_ref
        # else:
        #     delta = 1e-1
        #     y_ref = y_ref + delta
        #     p = np.diag(1/y_ref**2)
        #     q = -1/y_ref
        # regularisation term in the objective
        if obj_reg:
            assert isinstance(obj_reg, float), "If given 'obj_reg' must be a float (weight)."
            l = np.zeros((nh, nh))
            for i_h in range(nh):
                l[i_h, i_h] = 1
                l[i_h, (i_h + 1) % nh] = -1
            p += obj_reg * np.dot(l.T, l)  # q += np.zeros((nh,))
        # turn into cvxopt matrices
        P = opt.matrix(p)
        A = opt.matrix(a)
        b = opt.matrix(b)
        q = opt.matrix(q)
        G = opt.matrix(g)
        h = opt.matrix(h)
        # options for solver
        cvxopt = {**dict(kktsolver='ldl', options=dict(kktreg=1e-9, show_progress=False)), **cvxopt}
        # solve and retrieve solution
        sol = opt.solvers.qp(P=P, q=q, A=A, b=b, G=G, h=h, **cvxopt)
        self.status = Status(sol['status'])
        y = sol['x']
        y = np.array(y)
        return y.flatten()
