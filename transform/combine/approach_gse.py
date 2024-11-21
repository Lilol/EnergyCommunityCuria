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
import numpy as np

from data_storage.data_store import DataStore
from input.definitions import BillType
from transform.combine.methods_scaling import scale_gse
# self-created modules and functions
from transform.extract.utils import ProfileExtractor


# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
# common variables


# ----------------------------------------------------------------------------
#
def evaluate(bills, nds, pod_type, bill_type):
    if len(bills) != len(nds) or len(nds) != 12:
        raise ValueError("Length of input must match")
    #
    y = []
    for im, (bill, nd) in enumerate(zip(bills, nds)):
        y_ref = DataStore()["load_profiles"][(pod_type, im)]
        if bill_type == BillType.MONO:
            y_scale = y_ref / np.sum(ProfileExtractor.get_monthly_consumption(y_ref, nd)) * np.sum(bill)
        else:
            y_scale, _ = scale_gse(bill, nd, y_ref)
            if ((b := ProfileExtractor.get_monthly_consumption(y_scale, nd)) != bill).any():
                y_scale[np.isnan(y_scale)] = 0
                for if_, f in enumerate(fs):
                    # Just spread total consumption in F1 on F1 hours
                    y_scale[DataStore()["tariff"].flatten() == f] += (bill[if_] - (
                        b[if_] if not np.isnan(b[if_]) else 0)) / sum(
                        [np.count_nonzero(arera[j] == f) * nd[j] for j in js])
                if (np.abs(b - bill) > 0.1).any():
                    print("While correcting total consumption:")
                    print(f"True: {bill}")
                    print(f"Corr: {b}")
        y.append(y_scale)
    #
    return np.array(y)
