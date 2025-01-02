from input.definitions import DataKind


def eval_sc(df, n_fam):
    calculate_shared_energy(df, n_fam)
    return calculate_sc(df)


def find_closer(n_fam, step):
    """Return closer integer to n_fam, considering the step."""
    if n_fam % step == 0:
        return n_fam
    if n_fam % step >= step / 2:
        return (n_fam // step) + 1
    else:
        return n_fam // step


def find_best_value(df, n_fam_high, n_fam_low, step, sc):
    # Stopping criterion (considering that n_fam is integer)
    if n_fam_high - n_fam_low <= step:
        print("Procedure ended without exact match.")
        return n_fam_high, eval_sc(df, n_fam_high)

    # Bisection of the current space
    n_fam_mid = find_closer((n_fam_low + n_fam_high) / 2, step)
    sc_mid = eval_sc(df, n_fam_mid)

    # Evaluate and update
    if sc_mid - sc == 0:  # Check if exact match is found
        print("Found exact match.")
        return n_fam_mid, sc_mid

    elif sc_mid < sc:
        return find_best_value(df, n_fam_high, n_fam_mid, step, sc)
    else:
        return find_best_value(df, n_fam_mid, n_fam_low, step, sc)


def find_the_optimal_number_of_families_for_sc_ratio(df, sc, n_fam_max, step=25):
    """
    Finds the optimal number of families to satisfy a given self-consumption
    ratio.

    Parameters:
    - sc (float): Target self consumption ratio.
    - n_fam_max (int): Maximum number of families.
    - p_plants (numpy.ndarray): Array of power values from plants.
    - p_users (numpy.ndarray): Array of power values consumed by users.
    - p_fam (numpy.ndarray): Array of power values consumed by each family.
    - step (int): Step in the number of families.

    Returns:
    - tuple: Tuple containing the optimal number of families and the
        corresponding shared energy ratio.
    """

    # Evaluate starting point
    n_fam_low = 0
    sc_low = eval_sc(df, n_fam_low)
    if sc_low >= sc:  # Check if requirement is already satisfied
        print("Requirement already satisfied!")
        return n_fam_low, sc_low

    # Evaluate point that can be reached
    n_fam_high = n_fam_max
    sc_high = eval_sc(df, n_fam_high)
    if sc_high <= sc:  # Check if requirement is satisfied
        print("Requirement cannot be satisfied!")
        return n_fam_high, sc_high

    # Loop to find best value
    return find_best_value(df, n_fam_high, n_fam_low, step, sc)


def calc_sum_consumption(df, n_fam):
    df[DataKind.CONSUMPTION] = df[DataKind.CONSUMPTION_OF_FAMILIES] * n_fam + df[DataKind.CONSUMPTION_OF_USERS]


def calculate_shared_energy(df, n_fam):
    calc_sum_consumption(df, n_fam)
    df[DataKind.SHARED] = df[[DataKind.PRODUCTION, DataKind.CONSUMPTION]].min(axis="rows")


def calculate_sc(df):
    return df[DataKind.SHARED].sum() / df[DataKind.PRODUCTION].sum()
