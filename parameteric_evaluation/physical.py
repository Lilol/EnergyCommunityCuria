from abc import abstractmethod

import xarray as xr

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, OtherParameters
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class PhysicalParameterCalculator(Calculator):
    _key = PhysicalMetric.INVALID
    _name = "Physical evaluator"

    @classmethod
    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        pass

    @classmethod
    def postprocess(cls, result, results_of_previous_calculations: OmnesDataArray | None, parameters: dict):
        return results_of_previous_calculations


class SharedEnergy(PhysicalParameterCalculator):
    _key = PhysicalMetric.SHARED_ENERGY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        dx = input_da.sel(
            {DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]}).min(
            dim=DataKind.CALCULATED.value).assign_coords({DataKind.CALCULATED.value: cls._key})
        input_da = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return input_da, results_of_previous_calculations


class TotalConsumption(PhysicalParameterCalculator):
    _key = PhysicalMetric.TOTAL_CONSUMPTION

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        number_of_families = kwargs.get(DataKind.NUMBER_OF_FAMILIES.value)
        dx = (input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_FAMILIES}) * number_of_families + input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
            {DataKind.CALCULATED.value: cls._key})
        input_da = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return input_da, results_of_previous_calculations


class PhysicalMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.PHYSICAL_METRICS
    _name = "Physical metric evaluator"
