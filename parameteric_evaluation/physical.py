from typing import Iterable

import numpy as np

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation import MetricEvaluator
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, Parameter, Metric
from parameteric_evaluation.parametric_evaluation import ParametricEvaluator
from parameteric_evaluation.target_self_consumption import calculate_shared_energy, calculate_sc
from utility.dimensions import power_to_energy


class PhysicalParameterCalculator(Calculator):
    _parameter_calculated = Parameter.INVALID

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        pass


class SharedEnergy(PhysicalParameterCalculator):
    _parameter_calculated = Parameter.SHARED_ENERGY

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        return input_da.sel(data=[DataKind.P_INJ, DataKind.P_WITH]).min().sum()


class SelfConsumption(PhysicalParameterCalculator):
    _parameter_calculated = Metric.SC

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        return input_da.sel(data=DataKind.P_SH).sum() / input_da.sel(data=DataKind.P_INJ).sum()


class SelfSufficiency(PhysicalParameterCalculator):
    _parameter_calculated = Metric.SS

    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        return input_da.sel(data=DataKind.P_SH).sum() / input_da.sel(data=DataKind.P_WITH).sum()


class PhysicalParameterEvaluator(MetricEvaluator):
    _type = ParametricEvaluationType.PHYSICAL_METRICS
    _name = "physical_parameter_evaluation"
    _parameter_calculators = {Parameter.SELF_CONSUMPTION: SelfConsumption(),
                              Parameter.SELF_SUFFICIENCY: SelfSufficiency(), Parameter.SHARED_ENERGY: SharedEnergy()}

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        for parameter, calculator in self._parameter_calculators.items():
            calculator.calculate(dataset, dataset)
        return dataset

    @classmethod
    def eval_physical_parameters(cls, p_inj, p_with, dt=1, p_prod=None, p_cons=None):
        """
        Evaluates shared energy, produced energy, consumed energy, and related
        indicators.

        Parameters:
        - p_inj (numpy.ndarray): Array of injected power values.
        - p_with (numpy.ndarray): Array of withdrawn power values.
        - dt (float): Time interval (default is 1). In hours?
        - return_power (bool): Whether to return shared power values.
            Default is False.
        - p_prod (numpy.ndarray): Array of produced power values.
            If None, it is set equal to p_inj. Default is None
        - p_cons (numpy.ndarray): Array of consumed power values.
            If None, it is set equal to p_with. Default is None.

        Returns:
        - tuple: Tuple containing sc (shared energy ratio),
            ss (shared energy ratio), e_sh (shared energy).
        """

        # Get inputs
        p_prod = p_inj if p_prod is None else p_prod
        p_cons = p_with if p_cons is None else p_cons

        # Evaluate shared power
        p_sh = np.minimum(p_inj, p_with)

        # Evaluate energy quantities
        e_sh = power_to_energy(p_sh, dt=dt)  # shared energy
        e_prod = power_to_energy(p_prod, dt=dt)  # produced energy
        e_cons = power_to_energy(p_cons, dt=dt)  # consumed energy

        # Evaluate indicators
        sc = e_sh / e_prod  # shared energy to production ratio
        ss = e_sh / e_cons  # shared energy to consumption ratio
        e_inj = e_sh / sc  # power_to_energy(p_inj)?
        e_with = e_sh / ss  # power_to_energy(p_with)?

        # Return
        return sc, ss, e_sh, p_sh, e_inj, e_with

    @classmethod
    def calculate_theoretical_limit_of_self_consumption(cls, df_months, n_fam):
        calculate_shared_energy(df_months, n_fam)
        return calculate_sc(df_months)

    @classmethod
    def calculate_sc_for_time_aggregation(cls, df_hours, time_resolution, n_fam):
        """Evaluate self consumption with given temporal aggregation and number of families."""
        calculate_shared_energy(df_hours.groupby(time_resolution).sum(), n_fam)
        return calculate_sc(df_hours)
