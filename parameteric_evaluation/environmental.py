from pandas import read_csv

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind, ParametersFromFile
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, EnvironmentalMetric, PhysicalMetric, \
    OtherParameters
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from utility.configuration import config


class EmissionFactors(ParametersFromFile):
    _filename = config.get("parametric_evaluation", "emission_factors_configuration_file")

    @classmethod
    def read(cls, filename):
        values = read_csv(filename, index_col=0, header=0).to_dict()
        return values[list(values.keys())[0]]


class EmissionSavingsRatio(Calculator):
    _key = EnvironmentalMetric.ESR

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        # Evaluate emissions savings ratio
        em_base = results_of_previous_calculations.sel({DataKind.METRIC.value: EnvironmentalMetric.BASELINE_EMISSIONS})
        if em_base == 0:
            val = 0
        else:
            em_tot = results_of_previous_calculations.sel({DataKind.METRIC.value: EnvironmentalMetric.TOTAL_EMISSIONS})
            val = (em_base - em_tot) / em_base
        return input_da, val


class TotalEmissions(Calculator):
    _key = EnvironmentalMetric.TOTAL_EMISSIONS

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        """ Evaluate total emissions in REC case"""
        shared = input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum()
        total_emissions = (input_da.sel({DataKind.CALCULATED.value: OtherParameters.WITHDRAWN_ENERGY}).sum() - shared) * \
                          EmissionFactors()["grid"] + (input_da.sel(
            {DataKind.CALCULATED.value: OtherParameters.INJECTED_ENERGY}).sum() - shared) * EmissionFactors()[
                              "inj"] + input_da.sel({DataKind.CALCULATED.value: DataKind.PRODUCTION}).sum() * \
                          EmissionFactors()["prod"] * kwargs.get("years") + kwargs.get("bess_size") * EmissionFactors()[
                              "bess"]
        return input_da, total_emissions


class BaselineEmissions(Calculator):
    _key = EnvironmentalMetric.BASELINE_EMISSIONS

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args,
                  **kwargs) -> tuple[OmnesDataArray, float | None]:
        """ Evaluate total emissions in base case"""
        baseline_emissions = input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.TOTAL_CONSUMPTION}).sum() * EmissionFactors()[
            "grid"] * kwargs.get("years")
        return input_da, baseline_emissions


class EnvironmentalEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.ENVIRONMENTAL_METRICS
    _name = "Environmental evaluator"
    _years = config.getint("parametric_evaluation", "economic_evaluation_number_of_years")

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        """
        Calculates the CO2 emissions based on the shared energy, consumed energy,
        produced energy, and emission factors.
        """
        return super().invoke(*args, **kwargs, years=cls._years)
