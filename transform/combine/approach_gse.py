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
from transform.extract.utils import ProfileExtractor


def evaluate(bills, nds, pod_type, bill_type):
    y = []
    for im, (bill, nd) in enumerate(zip(bills, nds)):
        y_ref = DataStore()["typical_load_profiles_gse"][(pod_type, im)]
        if bill_type == BillType.MONO:
            y_scale = y_ref / np.sum(ProfileExtractor.get_monthly_consumption(y_ref)) * np.sum(bill)
        else:
            y_scale, _ = scale_gse(bill, y_ref)
        y.append(y_scale)
    return np.array(y)
