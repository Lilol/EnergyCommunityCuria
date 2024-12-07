# ----------------------------------------------------------------------------
# Import statement

# Data management
import os

# Data processing
import numpy as np
import pandas as pd

from utility import configuration
from input.definitions import DataKind
from utility.day_of_the_week import df_year


# ----------------------------------------------------------------------------
# Useful functions

# Calculate energy by summing up power values multiplied by the time interval
def energy(p, dt=1):
    """Calculates the total energy given power (p) and time step (dt)."""
    return p.sum() * dt


# Manage power flows when a battery is present
def manage_bess(p_prod, p_cons, bess_size, t_min=None):
    """Manage BESS power flows to increase shared energy."""

    # Get inputs
    p_bess_max = np.inf if t_min is None else bess_size / t_min

    # Initialize BESS power array and stored energy
    p_bess = np.zeros_like(p_prod)
    e_stored = 0

    # Manage flows in all time steps
    for h, (p, c) in enumerate(zip(p_prod, p_cons)):
        # Power to charge the BESS (discharge if negative)
        b = p - c
        # Correct according to technical/physical limits
        if b < 0:
            b = max(b, -e_stored, -p_bess_max)
        else:
            b = min(b, bess_size - e_stored, p_bess_max)
        # Update BESS power array and stored energy
        p_bess[h] = b
        e_stored = e_stored + b

    return p_bess


# Evaluate shared energy, produced energy, consumed energy, and related
# indicators
def eval_rec(p_inj, p_with, dt=1, return_power=False, p_prod=None, p_cons=None):
    """
    Evaluates shared energy, produced energy, consumed energy, and related
    indicators.

    Parameters:
    - p_inj (numpy.ndarray): Array of injected power values.
    - p_with (numpy.ndarray): Array of withdrawn power values.
    - dt (float): Time interval (default is 1).
    - return_power (bool): Whether to return shared power values.
        Default is False.
    - p_prod (numpy.ndarray): Array of produced power values.
        If None, it is set equal to p_inj. Default is None
    - p_cons (numpy.ndarray): Array of consumed power values.
        If None, it is set equal to p_with. Default is None.

    Returns:
    - tuple: Tuple containing sc (shared energy ratio),
        ss (shared energy ratio), e_sh (shared energy).
        If return_power=True, also returns p_sh (shared power values).
    """

    # Get inputs
    p_prod = p_inj if p_prod is None else p_prod
    p_cons = p_with if p_cons is None else p_cons

    # Evaluate shared power
    p_sh = np.minimum(p_inj, p_with)

    # Evaluate energy quantities
    e_sh = energy(p_sh, dt=dt)  # shared energy
    e_prod = energy(p_prod, dt=dt)  # produced energy
    e_cons = energy(p_cons, dt=dt)  # consumed energy

    # Evaluate indicators
    sc = e_sh / e_prod  # shared energy to production ratio
    ss = e_sh / e_cons  # shared energy to consumption ratio

    # Return
    if return_power:
        return sc, ss, e_sh, p_sh
    return sc, ss, e_sh


# Calculate the CO2 emissions
def eval_co2(e_sh, e_cons, e_prod, e_with=None, e_inj=None, bess_size=0, eps_grid=0.263, eps_inj=0, eps_prod=0.05,
             eps_bess=175, n=20):
    """
    Calculates the CO2 emissions based on the shared energy, consumed energy,
    produced energy, and emission factors.

    Parameters:
    - e_sh (float): Shared energy.
    - e_cons (float): Consumed energy.
    - e_prod (float): Produced energy.
    - bess_size (float): Size of Battery Energy Storage System (BESS) in kWh.
        Default is 0.
    - eps_grid (float): Emission factor for energy from the grid.
        Default is 0.263 kgCO2eq/kWh.
    - eps_inj (float): Emission factor for injected energy.
        Default is -0 kgCO2eq/kWh.
    - eps_prod (float): Emission factor for produced energy (LCA).
        Default is 0.05 kgCO2eq/kWh.
    - eps_bess (float): Emission factor for BESS capacity.
        Default is 175 kgCO2eq/kWh.
    - n (int): Number of years considered. Default is 20.

    Returns:
    - Tuple[float, float, float]: Emissions savings ratio, total emissions,
        and baseline emissions.
    """
    # Get values of injections and withdrawals
    e_inj = e_prod if e_inj is None else e_inj
    e_with = e_cons if e_with is None else e_with

    # Evaluate total emissions
    em_tot = ((e_with - e_sh) * eps_grid + (e_inj - e_sh) * eps_inj + eps_prod * e_prod) * n + bess_size * eps_bess

    # Evaluate total emissions in base case
    em_base = (e_cons * eps_grid) * n

    # Evaluate emissions savings ratio
    esr = (em_base - em_tot) / em_base

    return esr, em_tot, em_base


def eval_capex_pv(pv_size):
    """Evaluate investment cost (CAPEX) of a PV system depending on the size."""
    # if pv_size < 10 :
    #     capex_pv = 1900
    # elif pv_size < 35 :
    #     capex_pv = -6 * pv_size + 1960
    # elif pv_size < 125 :
    #     capex_pv = -7.2 * pv_size + 2002.8
    # elif pv_size < 600 :
    #     capex_pv = -0.74 * pv_size + 1192.1
    # else :
    #     capex_pv=750
    if pv_size < 20:
        c_pv = 1500
    elif pv_size < 200:
        c_pv = 1200
    elif pv_size < 600:
        c_pv = 1100
    else:
        c_pv = 1050
    return c_pv * pv_size


# Evaluate the investment cost for the REC
def eval_capex(pv_sizes, bess_size, n_users, c_bess=350, c_user=100):
    """Evaluate CAPEX of a REC, given PV sizes, BESS size(s) and
    number of users."""

    # Initialize CAPEX
    capex = 0

    # Add cost of PVS
    for pv_size in pv_sizes:
        capex += eval_capex_pv(pv_size)

    # Add cost of BESS
    capex += bess_size * c_bess

    # Add cost of users
    capex += n_users * c_user

    return capex
