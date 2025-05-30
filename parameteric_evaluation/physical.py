from abc import abstractmethod
from typing import Iterable

import xarray as xr

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, OtherParameters
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class PhysicalParameterCalculator(Calculator):
    _key = PhysicalMetric.INVALID

    @classmethod
    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        pass


class SharedEnergy(PhysicalParameterCalculator):
    _key = PhysicalMetric.SHARED_ENERGY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        dx = input_da.sel({DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY,
                                                       OtherParameters.WITHDRAWN_ENERGY]}).min().assign_coords(
            {DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        input_da = xr.concat([input_da, dx], dim=DataKind.METRIC.value)
        return input_da, output


class TotalConsumption(PhysicalParameterCalculator):
    _key = PhysicalMetric.TOTAL_CONSUMPTION

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        number_of_families = kwargs.get('number_of_families')
        dx = (input_da.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_FAMILIES}) * number_of_families + input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
            {DataKind.CALCULATED.value: cls._key})
        input_da = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return input_da, output


class PhysicalMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.PHYSICAL_METRICS
    _name = "physical_metric_evaluation"
