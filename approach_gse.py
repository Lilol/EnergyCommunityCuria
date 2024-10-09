"""
Module 'approach_gse.py'
____________
DESCRIPTION
______
NOTES
_____
INFO
Author: G. Lorenti (gianmarco.lorenti@polito.it)
Date: 30.01.2023
"""
# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
import pandas as pd
import numpy as np
# self-created modules and functions
from methods_scaling import scale_gse, scale_qopt
from utils import eval_x, eval_y_flat
# common variables
from common import *


# ----------------------------------------------------------------------------
#
def evaluate(bills, nds, pod_type, bill_type):
    """
    Function 'evaluate'
    ____________
    DESCRIPTION
    ______
    NOTES
    _____
    INFO
    Author: G. Lorenti (gianmarco.lorenti@polito.it)
    Date: 07.02.2023
    """
    #
    assert len(bills) == len(nds) == nm
    assert pod_type in ['bta', 'dom', 'ip']
    assert bill_type in ['mono', 'tou']
    #
    y = []
    for im, (bill, nd) in enumerate(zip(bills, nds)):
        y_ref = y_ref_gse[(pod_type, im)]
        if bill_type == 'mono':
            y_scale = y_ref / np.sum(eval_x(y_ref, nd)) * np.sum(bill)
        else:
            y_scale, _ = scale_gse(bill, nd, y_ref)
            if ((b := eval_x(y_scale, nd)) != bill).any():
                y_scale[np.isnan(y_scale)] = 0
                for if_, f in enumerate(fs):
                    # Just spread total consumption in F1 on F1 hours
                    y_scale[arera.flatten() == f] += \
                        (bill[if_] - (b[if_] if not np.isnan(b[if_]) else 0)) /\
                        sum([np.count_nonzero(arera[j] == f) * nd[j] for j in js])
                if (np.abs(eval_x(y_scale, nd) - bill) > 0.1).any():
                    print("While correcting total consumption:")
                    print("True:{}".format(bill))
                    print("Corr:{}".format(eval_x(y_scale, nd)))
        y.append(y_scale)
    #
    return np.array(y)
