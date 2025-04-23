from typing import Iterable

import cvxopt as opt
import numpy as np

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from operation import ScaleProfile
from operation.definitions import ScalingMethod, Status
from utility import configuration


class ScaleByQuadraticOptimization(ScaleProfile):
    _name = "quadratic_optimization_scaler"
    _key = ScalingMethod.QUADRATIC_OPTIMIZATION

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
        The method solves a quadratic optimization problem where the deviation
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
            Objective of the optimization
            0 : minimise sum(y[i] - y_ref[i])**2;
            1 : minimise sum(y[i]/y_ref[i] -1)**2.
            Default is 0.
        obj_reg : float or None, optional
            Weight of "regularisation" term to the objective
            Regularisation is intended as difference in the demand between
            subsequent time-steps, i.e.: obj_reg * sum(y[i] - y[(i+1)%nh])**2.
            Default is None, same as 0.
        cvxopt : dict, optional
            Parameters to pass to 'cvxopt'.
        _______
        RETURNS
        y : np.ndarray
            Estimated hourly load profile in each day-type
            Array of shape (nj*ni) where 'ni' is the number of time-steps in each
            day.
        status : str
            Status of the optimization
            Can be : 'optimal', 'unknown', 'unfeasible'.
        _____
        INFO
        Author : G. Lorenti (gianmarco.lorenti@polito.it)
        Date : 17.11.2022 (last update: 17.11.2022)
        """
        reference_profile = operands[1]
        total_consumption_by_time_slots = operands[2].fillna(0).sum(DataKind.DAY_TYPE.value)
        y_max = self.kwargs.pop("y_max", None)
        obj = self.kwargs.pop("obj", 0)
        obj_reg = self.kwargs.pop("obj_reg", None)
        cvxopt = self.kwargs.pop("cvxopt", {})
        # ------------------------------------
        # total consumption in the reference profile in the month
        # x_ref = np.array([
        #     sum([np.sum(y_ref.reshape(nj, ni)[ij, arera[ij]==f])*nd[ij]
        #           for ij in range(nj)]) for f in fs])
        # y_ref = y_ref * np.sum(x) / np.sum(x_ref)
        # ------------------------------------
        # constraint in the optimization problem
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
        ds = DataStore()
        time_of_use_time_slots = ds["time_of_use_time_slots"]
        number_of_days_in_month = ds["day_count"].sel({DataKind.MONTH.value: reference_profile.month.values})
        # constraint on the total consumption (one for each tariff time slot)
        for if_, f in enumerate(number_of_time_of_use_periods):
            aux = np.concatenate([(time_of_use_time_slots.sel(
                {DataKind.DAY_TYPE.value: day_type}).values == f) * number_of_days_in_month.sel(
                {DataKind.DAY_TYPE.value: day_type}) for day_type in
                                  configuration.config.getarray("time", "day_types", int)])
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
        nd = np.repeat((number_of_days_in_month / number_of_days_in_month.sum()).values, n_time_steps)
        p = np.diag(nd)
        q = -reference_profile * nd
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
