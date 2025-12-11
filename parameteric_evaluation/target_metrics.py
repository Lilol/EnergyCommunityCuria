import logging
from typing import override

import numpy as np
from pandas import DataFrame

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.output.write import WriteDataArray
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, LoadMatchingMetric
from parameteric_evaluation.load_matching_evaluation import SelfConsumption
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import SharedEnergy, PhysicalParameterCalculator
from utility import configuration

logger = logging.getLogger(__name__)


class TargetMetricParameterCalculator(Calculator):
    _key = None
    _metric = LoadMatchingMetric.INVALID
    _param_calculator = PhysicalParameterCalculator.create(_metric)

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]: ...

    def eval(self, df, n_fam):
        self._param_calculator.calculate(df, num_families=n_fam)
        return SelfConsumption.calculate(df)

    @staticmethod
    def find_closer(n_fam, step):
        """Return closer integer to n_fam, considering the step."""
        if n_fam % step == 0:
            return n_fam
        if n_fam % step >= step / 2:
            return (n_fam // step) + 1
        else:
            return n_fam // step

    def find_best_value(self, df, n_fam_high, n_fam_low, step, current_value):
        # Stopping criterion (considering that n_fam is integer)
        if n_fam_high - n_fam_low <= step:
            print("Procedure ended without exact match.")
            return n_fam_high, eval(df, n_fam_high)

        # Bisection of the current space
        n_fam_mid = self.find_closer((n_fam_low + n_fam_high) / 2, step)
        mid = eval(df, n_fam_mid)

        # Evaluate and update
        if mid - current_value == 0:  # Check if exact match is found
            print("Found exact match.")
            return n_fam_mid, mid

        elif mid < current_value:
            return self.find_best_value(df, n_fam_high, n_fam_mid, step, current_value)
        else:
            return self.find_best_value(df, n_fam_mid, n_fam_low, step, current_value)

    def find_the_optimal_number_of_families_for_value(self, df, val, n_fam_max, step=25):
        """
        Finds the optimal number of families to satisfy a given self-consumption
        ratio.

        Parameters:
        - val (float): Target metric ratio.
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
        low = self.eval(df, n_fam_low)
        if low >= val:  # Check if requirement is already satisfied
            print("Requirement already satisfied!")
            return n_fam_low, low

        # Evaluate point that can be reached
        n_fam_high = n_fam_max
        high = self.eval(df, n_fam_high)
        if high <= val:  # Check if requirement is satisfied
            print("Requirement cannot be satisfied!")
            return n_fam_high, high

        # Loop to find best value
        return self.find_best_value(df, n_fam_high, n_fam_low, step, val)


# Dynamically create calculator subclasses
for metric in LoadMatchingMetric:
    if metric == LoadMatchingMetric.INVALID:
        continue

    class_name = f"{metric.value.replace(' ', '')}TargetCalculator"
    def make_calculator(m):
        class _Calc(TargetMetricParameterCalculator):
            _key = m
            _metric = m
            _param_calculator = PhysicalParameterCalculator.create(_metric)

            @classmethod
            def calculate(cls, input_da: OmnesDataArray | None = None,
                          results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
                OmnesDataArray, float | None]:
                """Evaluate the number of households that is needed to reach the specific value of a load matching metric."""
                aggregated = input_da.groupby(
                        f"time.{cls._aggregation.value}").sum()
                aggregated, _ = SharedEnergy.calculate(aggregated)
                return input_da, cls._param_calculator.calculate(aggregated)[1]

        _Calc.__name__ = class_name
        return _Calc

    globals()[class_name] = make_calculator(metric)


class TargetMetricEvaluator(ParametricEvaluator):
    _name = "Target metric evaluator"
    _key = ParametricEvaluationType.METRIC_TARGETS
    _metric = LoadMatchingMetric.INVALID
    _max_number_of_households = configuration.config.getint("parametric_evaluation", "max_number_of_households")

    @classmethod
    @override
    def invoke(cls, *args, **kwargs):
        input_da, results = cls.evaluate_targets(*args, **kwargs)
        WriteDataArray().execute(results, attribute="target_metrics")
        return input_da, results

    @staticmethod
    def get_eval_metrics(evaluation_type):
        return {m: TargetMetricParameterCalculator.get_subclass(m) for m in LoadMatchingMetric if
                m != LoadMatchingMetric.INVALID}

    @classmethod
    def get_targets(cls):
        return configuration.config.getarray("parametric_evaluation",
                                             f"{cls._metric.value.lower().replace(' ', '_')}_targets", float)

    @classmethod
    def evaluate_targets(cls):
        targets = cls.get_targets()

        results = DataFrame(np.nan, index=targets, columns=["number_of_families", "metric_realized"])
        # Evaluate number of families for each target
        val, nf = 0, 0
        for target in targets:
            # # Skip if previous target was already higher than this
            if val >= target:
                results.loc[target, ["number_of_families", "metric_realized"]] = nf, val
                continue

            # # Find number of families to reach target
            nf, val = find_the_optimal_number_of_families_for_value(DataStore()["energy_year"], target,
                                                                    cls._max_number_of_households)

            # Update
            results.loc[target, ["number_of_families", "metric_realized"]] = nf, val

            # # Exit if targets cannot be reached
            if val < target:
                logger.warning(f"Exiting loop because '{target}' cannot be reached.")
                break
            if nf >= cls._max_number_of_households:
                logger.warning(f"Exiting loop because max families ({cls._max_number_of_households}) was reached.")
                break
        logger.info(
            f"Targets reached; number of families:\n{','.join(f'{row.number_of_families}; {row.self_consumption_realized}' for _, row in results.iterrows())}")
        logger.info(f"To provide battery sizes for evaluation in the configuration file use:\n[parametric_evaluation]"
                    f"\nevaluation_parameters={{'bess_sizes': [...], 'number_of_families': [{f','.join(f'{nf}' for nf in results.number_of_families)}]}}\nor\n"
                    f"evaluation_parameters = {{'number_of_families': {f','.join(f'{nf}: [...]' for nf in results.number_of_families)}}}")
        return results
