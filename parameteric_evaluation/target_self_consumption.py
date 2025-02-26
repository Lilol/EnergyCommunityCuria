import logging

import numpy as np
from pandas import DataFrame

from data_storage.data_store import DataStore
from parameteric_evaluation.definitions import ParametricEvaluator, ParametricEvaluationType
from parameteric_evaluation.time_aggregation_evaluation import calculate_shared_energy, calculate_sc
from utility import configuration

logger = logging.getLogger(__name__)


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


class TargetSelfConsumptionEvaluator(ParametricEvaluator):
    _type = ParametricEvaluationType.SELF_CONSUMPTION_TARGETS

    @classmethod
    def evaluate_self_consumption_targets(cls):
        self_consumption_targets = configuration.config.getarray("parametric_evaluation", "self_consumption_targets")
        max_number_of_households = configuration.config.getint("parametric_evaluation", "max_number_of_households")
        df = DataFrame(np.nan, index=self_consumption_targets,
                       columns=["number_of_families", "self_consumption_realized"])
        # Evaluate number of families for each target
        sc, nf = 0, 0
        for sc_target in self_consumption_targets:
            # # Skip if previous target was already higher than this
            if sc >= sc_target:
                df.loc[sc_target, ["number_of_families", "self_consumption_realized"]] = nf, sc
                continue

            # # Find number of families to reach target
            nf, sc = find_the_optimal_number_of_families_for_sc_ratio(DataStore()["energy_year"], sc_target,
                                                                      max_number_of_households)

            # Update
            df.loc[sc_target, ["number_of_families", "self_consumption_realized"]] = nf, sc

            # # Exit if targets cannot be reached
            if sc < sc_target:
                logger.warning("Exiting loop because requirement cannot be reached.")
                break
            if nf >= max_number_of_households:
                logger.warning("Exiting loop because max families was reached.")
                break
        logger.info(
            f"Self consumption targets reached; number of families:\n{','.join(f'{row.number_of_families}; {row.self_consumption_realized}' for _, row in df.iterrows())}")
        logger.info(f"To provide battery sizes for evaluation in the configuration file use:\n[parametric_evaluation]"
                    f"\nevaluation_parameters={{'bess_sizes': [...], 'number_of_families': [{f','.join(f'{nf}' for nf in df.number_of_families)}]}}\nor\n"
                    f"evaluation_parameters = {{'number_of_families': {f','.join(f'{nf}: [...]' for nf in df.number_of_families)}}}")
        logger.info(f"For battery evaluation please set \n[parametric_evaluation]\nuse_bess=True")
        return df
