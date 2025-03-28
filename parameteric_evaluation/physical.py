from abc import abstractmethod
from typing import Iterable

import numpy as np
import xarray as xr

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric
from parameteric_evaluation.dimensions import power_to_energy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class PhysicalParameterCalculator(Calculator):
    _key = PhysicalMetric.INVALID

    @classmethod
    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        pass


class SharedEnergy(PhysicalParameterCalculator):
    _key = PhysicalMetric.SHARED_ENERGY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        dx = input_da.sel({DataKind.CALCULATED.value: [PhysicalMetric.INJECTED_ENERGY,
                                                       PhysicalMetric.WITHDRAWN_ENERGY]}).min().assign_coords(
            {DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        data = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return data


class ConsumptionOfFamilies(PhysicalParameterCalculator):
    _key = PhysicalMetric.TOTAL_CONSUMPTION

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        num_families = kwargs.get('num_families')
        dx = (input_da.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_FAMILIES}) * num_families + input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION})
        input_da = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return input_da


class Equality(PhysicalParameterCalculator):
    _key = PhysicalMetric.INVALID
    _data_kind_to_derive_from = None

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        new_coords = input_da.coords[DataKind.CALCULATED.value]
        new_coords[new_coords == cls._data_kind_to_derive_from] = cls._key
        return input_da.assign_coords({DataKind.CALCULATED.value: (DataKind.CALCULATED.value, new_coords)})


class InjectedEnergy(Equality):
    _key = PhysicalMetric.INJECTED_ENERGY
    _data_kind_to_derive_from = DataKind.PRODUCTION


class WithdrawnEnergy(Equality):
    _key = PhysicalMetric.WITHDRAWN_ENERGY
    _data_kind_to_derive_from = DataKind.CONSUMPTION


class PhysicalMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.PHYSICAL_METRICS
    _name = "physical_metric_evaluation"

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
        e_inj = e_sh / sc  # TODO: power_to_energy(p_inj)?
        e_with = e_sh / ss  # TODO: power_to_energy(p_with)?

        # Return
        return sc, ss, e_sh, p_sh, e_inj, e_with
