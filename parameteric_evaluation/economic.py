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


def eval_capex(pv_sizes, bess_size, n_users, c_bess=350, c_user=100):
    """Evaluate CAPEX of a REC, given PV sizes, BESS size(s) and number of users."""

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
