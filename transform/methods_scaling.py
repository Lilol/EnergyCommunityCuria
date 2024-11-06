# -*- coding: utf-8 -*-
"""
Module 'methods_scaling.py'
____________
DESCRIPTION
______
NOTES
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : 17.11.2022 (last update : 17.11.2022)
"""
import cvxopt as opt
# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
import numpy as np
import numpy.linalg as lin


# ----------------------------------------------------------------------------
# 2. METHODS TO SCALE HOURLY LOAD PROFILES IN TYPICAL DAYS TO MONTHLY 
# ENERGY CONSUMPTION
# ----------------------------------------------------------------------------
# 2.1 Method 'gse tariff timeslots'
def scale_gse(x, nd, y_ref):
    """
    Function 'scale_gse'
    ____________
    DESCRIPTION
    The function evaluates hourly load profiles (y_scale) in each type of 
    day (j) scaling given reference load profiles (y_ref) in order to respect 
    the monthly energy consumption divided into tariff time-slots (x).
    ______
    NOTES
    The method evaluates one scaling factor for each tariff time-slot, which 
    is equal to the actual monthly consumption in that tariff time-slot
    divided by the consumption associated with the reference load profile. 
    The latter is then scaled separately for the time-steps in each time-slot.
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
        Can be : 'ok', 'unphysical', 'error'. No, not really.
    _____
    INFO
    Author : G. Lorenti (gianmarco.lorenti@polito.it)
    Date : 29.11.2022 (last update: 29.11.2022)
    """
    # ------------------------------------
    # check consistency of data
    # division of 'x' into tariff time slots
    assert (size := x.size) == nf, f"'x' must have size {nf}, not {size}."
    # division of 'n_days' into day-types
    assert (size := nd.size) == nj, f"'nd' must have size {nj}, not {size}."
    # total number of time-steps in 'y_ref'
    assert (size := np.size(y_ref)) == ni * nj, f"'y_ref' must have size {ni * nj}, not {size}."
    # ------------------------------------
    # scale reference profiles
    # evaluate the monthly consumption associated with the reference profile
    # divided into tariff time-slots
    x_ref = get_monthly_consumption(y_ref, nd)
    # calculate scaling factors k (one for each tariff time-slot)
    k_scale = x / x_ref
    # evaluate load profiles by scaling the reference profiles
    y_scal = y_ref.copy()
    # time-steps belonging to each tariff time-slot are scaled separately
    for if_, f in enumerate(fs):
        y_scal[arera.flatten() == f] = y_ref[arera.flatten() == f] * k_scale[if_]
    # ---------------------------------------
    # return
    return y_scal, 'optimal'


# ----------------------------------------------------------------------------
# 2.2 Method 'scale set of equations'
def scale_seteq(x, nd, y_ref):
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
    _____
    INFO
    Author : G. Lorenti (gianmarco.lorenti@polito.it)
    Date : 17.11.2022 (last update: 17.11.2022)
    """
    # ------------------------------------
    # check consistency of data
    # division of 'x' into tariff time slots
    assert (size := x.size) == nf, f"'x' must have size {nf}, not {size}."
    # division of 'n_days' into day-types
    assert (size := nd.size) == nj, f"'nd' must have size {nj}, not {size}."
    # total number of time-steps in 'y_ref'
    assert (size := np.size(y_ref)) == ni * nj, f"'y_ref' must have size {ni * nj}, not {size}."
    # ------------------------------------
    # evaluate scaling factors to get load profiles
    y_ref = y_ref.reshape(nj, ni)
    # energy consumed in reference profiles assigned to each typical day,
    # divided in tariff time-slots
    e_ref = np.concatenate(
        [np.array([[y_ref[ij, arera[ij] == f].sum() * nd[ij] for ij in range(nj)]]) for _, f in enumerate(fs)], axis=0)
    # scaling factors to respect total consumption
    status = 'optimal'
    try:
        k = np.dot(lin.inv(e_ref), x[:, np.newaxis])
        if np.any(k < 0):
            status = 'unphysical'
    except lin.LinAlgError:
        k = -1
        status = 'error'
    # ------------------------------------
    # get load profiles in day-types and return
    y_scal = y_ref * k.flatten()[:, np.newaxis]
    return y_scal.flatten(), status


# ----------------------------------------------------------------------------
# 2.3 Method "optimization"
# This method evaluates the load profile using a simple quadratic optimization
# problem, where the sum of the deviations over subsequent time steps is to be
# minimised. The total consumption is set as a constraint. 
# Moreover, other constraints can be added.
def scale_qopt(x, nd, y_ref, y_max=None, obj=0, obj_reg=None, cvxopt=None):
    """
    Function 'scale_qopt'
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
    # ------------------------------------
    # check consistency of data
    # division of 'x' into tariff time slots
    assert (size := x.size) == nf, f"'x' must have size {nf}, not {size}."
    # division of 'nd' into day-types
    assert (size := nd.size) == nj, f"'nd' must have size {nj}, not {size}."
    # total number of time-steps in 'y_ref'
    assert (size := np.size(y_ref)) == ni * nj, f"'y_ref' must have size {ni * nj}, not {size}."
    if cvxopt is None:
        cvxopt = {}
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
    status = sol['status']
    y = sol['x']
    y = np.array(y)
    return y.flatten(), status


#################################### TEST ####################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -------------------------------------
    # setup
    # energy consumption in one month divided in tariff time-slots (kWh)
    x = np.array([151.8, 130.3, 127.1])  # sport centre
    x = np.array([191.4, 73.8, 114.55])  # office
    x = np.array([200, 100, 300])
    # number of days of each day-type in the month
    nd = np.array([22, 4, 5])
    # maximum demand allowed
    y_max = 1.25
    # time-step on which to "anchor" load profiles in each day-type
    i_anch = 0
    # options for methods that scale a reference profile
    #  assigned
    ref = 'office'
    y_ref_db = dict(office=np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.4, 0.65, 0.9, 0.95, 1, 1, 0.95, 0.9, 0.85, 0.65, 0.45, 0.4, 0.35,
         0.25, 0.25, 0.25, 0.25]), sport_centre=np.array(
        [0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.45, 0.55, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.85, 1, 1, 1,
         0.95, 0.75, 0.55]), )
    y_ref = np.repeat(y_ref_db[ref][np.newaxis, :], 3, axis=0)
    y_ref = y_ref.flatten()
    # options for 'q_opt' method
    qopt_obj = 0
    qopt_obj_reg = 0
    # ------------------------------------
    # methods
    # method 'scale_gse'
    y_gse, _ = scale_gse(x, nd, y_ref)
    # print ("Scale, set_eq {}: ".format(ref), stat)
    # method 'scale_seteq'
    y_seteq, stat = scale_seteq(x, nd, y_ref)
    print(f"Scale, set_eq {ref}: ", stat)
    # method 'scale_qopt'
    y_qopt, stat = scale_qopt(x, nd, y_ref, y_max=y_max, obj=qopt_obj, obj_reg=qopt_obj_reg)
    print(f"Scale, q_opt {ref}: ", stat)
    # ------------------------------------
    # plot
    # figure settings
    figsize = (20, 10)
    fontsize = 25
    f_styles = [dict(color='red', alpha=0.1), dict(color='yellow', alpha=0.1), dict(color='green', alpha=0.1)]
    fig, ax = plt.subplots(figsize=figsize)
    time = np.arange(nh) + 0.5 * (dt := 24 / ni)
    # plot profiles
    ax.plot(time, y_gse, label='Scale, gse', marker='s', lw=2, ls='-', )
    ax.plot(time, y_seteq, label='Scale, seteq', marker='s', lw=2, ls='-', )
    ax.plot(time, y_qopt, label='Scale, qopt', marker='s', lw=2, ls='-', )
    # plot reference profile
    if ref:
        ax.plot(time, y_ref, color='k', ls='--', lw=1, label='Ref')
    # plot line for 'y_max'
    if y_max:
        ax.axhline(y_max, color='red', ls='--', lw=1, label='y max')
    # plot division into tariff time-slots
    f_sw_pos = []
    f_sw_styles = []
    h0 = 0
    for h in range(1, nh + 1):
        if h >= nh or arera.flatten()[h] != arera.flatten()[h0]:
            f_sw_pos.append((h0, h))
            f_sw_styles.append(f_styles[fs.index(arera.flatten()[h0])])
            h0 = h
    for pos, style in zip(f_sw_pos, f_sw_styles):
        ax.axvspan(*pos, **style, )
    for h in range(0, nh, ni):
        ax.axvline(h, color='grey', ls='-', lw=1)  # ax settings
    ax.set_xlabel("Time (h)", fontsize=fontsize)
    ax.set_ylabel("Power (kW)", fontsize=fontsize)
    ax.set_xticks((np.append(time, time[-1] + dt) - dt / 2)[::4])
    ax.set_xlim([0, time[-1] + dt / 2])
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, bbox_to_anchor=(1.005, 0.5), loc="center left")
    ax.grid(axis='y')
    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.8)
    plt.show()
    plt.close(fig)
    # fig.savefig('test_methods_scaling.png', dpi=300)
