from typing import Iterable

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation import MetricEvaluator
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType


class EmissionSavingsRatio(Calculator):
    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        # Evaluate emissions savings ratio
        em_base = kwargs.pop("em_base")
        return (em_base - kwargs.pop("em_tot")) / em_base


class TotalEmissions(Calculator):
    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        """ Evaluate total emissions
        - eps_grid (float): Emission factor for energy from the grid.
            Default is 0.263 kgCO2eq/kWh.
        - eps_inj (float): Emission factor for injected energy.
            Default is -0 kgCO2eq/kWh.
        - eps_prod (float): Emission factor for produced energy (LCA).
            Default is 0.05 kgCO2eq/kWh.
        - eps_bess (float): Emission factor for BESS capacity.
            Default is 175 kgCO2eq/kWh.
        - years (int): Number of years considered. Default is 20."""
        shared = input_da.sel(data=DataKind.SHARED)
        return (input_da.sel(data=DataKind.WITHDRAWN) - shared) * kwargs.pop("eps_grid", 0.263) + (
                input_da.sel(data=DataKind.INJECTED) - shared) * kwargs.pop("eps_inj", 0) + kwargs.pop("eps_prod",
                                                                                                       0.05) * input_da.sel(
            data=DataKind.PRODUCTION) * kwargs.pop("years", 20) + kwargs.pop("bess_size") * kwargs.pop("eps_bess", 175)


class BaselineEmissions(Calculator):
    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        """ Evaluate total emissions in base case
        - eps_grid (float): Emission factor for energy from the grid.
            Default is 0.263 kgCO2eq/kWh.
        - years (int): Number of years considered. Default is 20."""
        return (input_da.sel(data=DataKind.CONSUMPTION) * kwargs.pop("eps_grid", 0.263)) * kwargs.pop("years", 20)


class EnvironmentalEvaluator(MetricEvaluator):
    _type = ParametricEvaluationType.ENVIRONMENTAL_METRICS

    def invoke(self, *args, **kwargs):
        """
        Calculates the CO2 emissions based on the shared energy, consumed energy,
        produced energy, and emission factors.
        Returns:
        - Tuple[float, float, float]: Emissions savings ratio, total emissions,
            and baseline emissions as a DataArray.
        """
        baseline_emissions = BaselineEmissions.calculate(*args, **kwargs)
        total_emissions = TotalEmissions.calculate(*args, **kwargs)

        return baseline_emissions, total_emissions, EmissionSavingsRatio.calculate(*args, **kwargs,
                                                                                   em_tot=total_emissions,
                                                                                   em_base=baseline_emissions)
