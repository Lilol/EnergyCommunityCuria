from typing import Iterable

from data_storage.dataset import OmnesDataArray
from io_operation.input import DataKind, ParametersFromFile
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, EnvironmentalMetric, PhysicalMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from utility.configuration import config


class EmissionFactors(ParametersFromFile):
    _filename = config.get("parametric_evaluation", "emission_factors_configuration_file")


class EmissionSavingsRatio(Calculator):
    _key = EnvironmentalMetric.ESR

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        # Evaluate emissions savings ratio
        em_base = kwargs.pop("em_base")
        return (em_base - kwargs.pop("em_tot")) / em_base


class TotalEmissions(Calculator):
    _key = EnvironmentalMetric.TOTAL_EMISSIONS

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        """ Evaluate total emissions in REC case"""
        shared = input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        return (input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.WITHDRAWN_ENERGY}) - shared) * \
            EmissionFactors()["grid"] + (
                    input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.INJECTED_ENERGY}) - shared) * \
            EmissionFactors()["inj"] + input_da.sel({DataKind.CALCULATED.value: DataKind.PRODUCTION}) * \
            EmissionFactors()["prod"] * kwargs.get("years") + kwargs.get("bess_size") * EmissionFactors()["bess"]


class BaselineEmissions(Calculator):
    _key = EnvironmentalMetric.BASELINE_EMISSIONS

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        """ Evaluate total emissions in base case"""
        return input_da.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION}) * EmissionFactors()["gird"] * kwargs.get(
            "years")


class EnvironmentalEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.ENVIRONMENTAL_METRICS
    _name = "environmental_evaluator"
    _years = config.getint("parametric_evaluation", "economic_evaluation_number_of_years")

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        """
        Calculates the CO2 emissions based on the shared energy, consumed energy,
        produced energy, and emission factors.
        """
        results = kwargs.pop("results", args[0])
        baseline_emissions = BaselineEmissions.calculate(*args, **kwargs, years=cls._years)
        total_emissions = TotalEmissions.calculate(*args, **kwargs, years=cls._years)

        results.loc[results.index[-1], [m.to_abbrev_str() for m in cls._parameter_calculators]] = (
            baseline_emissions, total_emissions,
            EmissionSavingsRatio.calculate(*args, **kwargs, em_tot=total_emissions, em_base=baseline_emissions,
                                           years=cls._years))
