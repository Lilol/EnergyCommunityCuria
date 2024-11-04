# Reshape array of one-year data by days
import numpy as np
from pandas import Timestamp


def reshape_array_by_year(array, year):
    num_days_in_year = Timestamp(year, 12, 31).day_of_year

    # Calculate k based on the size of the array
    k = len(array) // num_days_in_year

    # Assert that k is an integer
    if len(array) % num_days_in_year != 0:
        raise ValueError("Cannot reshape array")

    # Reshape the array based on the number of days in the year and k
    return np.reshape(array, (num_days_in_year, k))
