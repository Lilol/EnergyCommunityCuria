# -*- coding: utf-8 -*-
"""
Module 'utils.py'
____________
DESCRIPTION
______
NOTES
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : 29.11.2022
"""
# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
# common variables
from common import *
# ----------------------------------------------------------------------------
# Method to evaluate monthly consumption from hourly load profiles
# evaluate the monthly consumption divided into tariff time-slots from the
# hourly load profiles in the day-types
def eval_x(y, nd):
    """
    Function 'eval_x'
    ____________
    DESCRIPTION
    The function evaluates the monthly energy consumption divided into tariff
    time-slots (x) given the hourly load profiles (y) in each type of day (j).
    ______
    NOTES
    ____________
    PARAMETERS
    y : np.ndarray
        Hourly load profile in each day-type
        Array of shape (nj*ni) where 'ni' is the number of time-steps in each
        day.
    nd : np.ndarray
        Number of days of each day-type in the month
        Array of shape (nj, ) where 'nj' is the number of day-types
        (according to ARERA's subdivision into day-types).
    _______
    RETURNS
    x : np.ndarray
        Monthly electricity consumption divided into tariff time-slots 
        Array of shape (nf, ) where 'nf' is the number of tariff time-slots.
    _____
    INFO
    Author : G. Lorenti (gianmarco.lorenti@polito.it)
    Date : 29.11.2022
    """
    # evaluate and return
    x = np.array([
        sum([(y.reshape(nj, ni)[j, arera[j] == f] * nd[j]).sum()
             for j in range(nj)]) for f in fs])
    return x
# ----------------------------------------------------------------------------
# This methods just spreads the total consumption over the tariff time-slots
# according to the number of hours of each of them
def eval_y_flat(x, nd):
    """
    Function 'flat'
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
    x : np.ndarray
        Monthly electricity consumption divided into tariff time-slots
        Array of shape (nf, ) where 'nf' is the number of tariff time-slots.
    nd : np.ndarray
        Number of days of each day-type in the month
        Array of shape (nj, ) where 'nj' is the number of day-types
        (according to ARERA's subdivision into day-types).
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
    # check consistency of data
    # division of 'x' into tariff time-slots
    assert (size := x.size) == nf, \
        f"'x' must have size {nf}, not {size}."
    # division of 'nd' into day-types
    assert (size := nd.size) == nj, \
        f"'nd' must have size {nj}, not {size}."
    # ------------------------------------
    # evaluate load profiles
    # count hours of each tariff time-slot in each day-type
    n_hours= np.array([np.count_nonzero(arera==f, axis=1) for f in fs])
    # count hours of each tariff time-slot in the month
    n_hours = np.sum(nd * n_hours, axis=1)
    # evaluate demand (flat) in each tariff time-slot
    k = x / n_hours
    # evaluate load profile in each day-type assigning to each time-step the
    # related demand, according to ARERA's profiles
    y = np.zeros_like(arera, dtype=float)
    for if_, f in enumerate(fs):
        y[arera == f] = k[if_]
    # ------------------------------------
    # return
    return y.flatten()


# ----------------------------------------------------------------------------
# Extract typical load profiles from year-long profile
def eval_y_from_year(p, months, day_types):
    assert len(p) == len(months) == len(day_types)
    y = []
    for im, m in enumerate(ms):
        for ij, j in enumerate(js):
            inds = ((months == m) & (day_types == j))
            ps = p[inds]
            assert (n := (len(ps) / ni)) == int(n) != 0
            ps = ps.reshape(int(n), ni)
            y.append(ps.mean(axis=0))
    return np.array(y)

# ----------------------------------------------------------------------------
# Extract typical load profiles from month-long profile
def eval_y_from_month(p, day_types):
    assert len(p) == len(day_types)
    y = []
    for ij, j in enumerate(js):
        inds = (day_types == j)
        ps = p[inds]
        assert (n := (len(ps) / ni)) == int(n) != 0
        ps = ps.reshape(int(n), ni)
        y.append(ps.mean(axis=0))
    return np.array(y)
