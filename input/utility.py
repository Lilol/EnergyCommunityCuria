# Reshape array of one-year data by days
import numpy as np


def reshape_array_by_year(array, year):
    """Reshapes an array to (number of days in the year, k), accounting for
    leap years."""

    # Check if the given year is a leap year
    is_leap_year = ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)

    # Get the number of days in the year based on whether it's a leap year
    num_days_in_year = 366 if is_leap_year else 365

    # Calculate k based on the size of the array
    k = len(array) // num_days_in_year

    # Assert that k is an integer
    assert isinstance(k, int), "The calculated value of k must be an integer."

    # Reshape the array based on the number of days in the year and k
    reshaped_array = np.reshape(array, (num_days_in_year, k))

    return reshaped_array
