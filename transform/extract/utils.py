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
import numpy as np

from data_storage.data_store import DataStore
from transform.extract.data_extractor import DataExtractor


class ProfileExtractor(DataExtractor):
    # ----------------------------------------------------------------------------
    # This methods spreads the total consumption over the tariff time-slots
    # according to the number of hours of each of them
    # TODO: this is analysis
    @staticmethod
    def spread_const_consumption_over_time_slots(total_consumption_by_tariff_slots, number_of_days_by_type):
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
        # evaluate load profiles
        # count hours of each tariff time-slot in each day-type
        n_hours = np.array([np.count_nonzero(arera == f, axis=1) for f in fs])
        # count hours of each tariff time-slot in the month
        n_hours = np.sum(number_of_days_by_type * n_hours, axis=1)
        # evaluate demand (flat) in each tariff time-slot
        k = total_consumption_by_tariff_slots / n_hours
        # evaluate load profile in each day-type assigning to each time-step the related demand, according to ARERA's profiles
        y = np.zeros_like(arera, dtype=float)
        for if_, f in enumerate(fs):
            y[arera == f] = k[if_]
        # ------------------------------------
        # return
        return y.flatten()

