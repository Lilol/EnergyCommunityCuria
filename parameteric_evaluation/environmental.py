from typing import Iterable

from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, EnvironmentalMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class EmissionSavingsRatio(Calculator):
    _key = EnvironmentalMetric.ESR

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        # Evaluate emissions savings ratio
        em_base = kwargs.pop("em_base")
        return (em_base - kwargs.pop("em_tot")) / em_base


class TotalEmissions(Calculator):
    _key = EnvironmentalMetric.TOTAL_EMISSIONS

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
    _key = EnvironmentalMetric.BASELINE_EMISSIONS

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        """ Evaluate total emissions in base case
        - eps_grid (float): Emission factor for energy from the grid.
            Default is 0.263 kgCO2eq/kWh.
        - years (int): Number of years considered. Default is 20."""
        return (input_da.sel(data=DataKind.CONSUMPTION) * kwargs.pop("eps_grid", 0.263)) * kwargs.pop("years", 20)


class EnvironmentalEvaluator(ParametricEvaluator):
    _type = ParametricEvaluationType.ENVIRONMENTAL_METRICS
    _name = "environmental_evaluator"
    _parameter_calculators = {EnvironmentalMetric.ESR: EmissionSavingsRatio(),
                              EnvironmentalMetric.BASELINE_EMISSIONS: BaselineEmissions(),
                              EnvironmentalMetric.TOTAL_EMISSIONS: TotalEmissions()}

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        """
        Calculates the CO2 emissions based on the shared energy, consumed energy,
        produced energy, and emission factors.
        """
        results = kwargs.pop("results", args[0])
        baseline_emissions = BaselineEmissions.calculate(*args, **kwargs)
        total_emissions = TotalEmissions.calculate(*args, **kwargs)

        results.loc[results.index[-1], [m.to_abbrev_str() for m in cls._parameter_calculators]] = (
        baseline_emissions, total_emissions,
        EmissionSavingsRatio.calculate(*args, **kwargs, em_tot=total_emissions, em_base=baseline_emissions))
