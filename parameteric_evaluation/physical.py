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
        dx = input_da.sel({DataKind.METRIC.value: [OtherParameters.INJECTED_ENERGY,
                                                       OtherParameters.WITHDRAWN_ENERGY]}).min().assign_coords(
            {DataKind.METRIC.value: PhysicalMetric.SHARED_ENERGY})
        output = xr.concat([output, dx], dim=DataKind.METRIC.value)
        return output, output.sel({DataKind.METRIC.value: PhysicalMetric.SHARED_ENERGY}).sum()


class TotalConsumption(PhysicalParameterCalculator):
    _key = PhysicalMetric.TOTAL_CONSUMPTION

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        num_families = kwargs.get('num_families')
        dx = (input_da.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_FAMILIES}) * num_families + input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
            {DataKind.METRIC.value: DataKind.CONSUMPTION})
        output = xr.concat([output, dx], dim=DataKind.METRIC.value)
        return output, output.sel({DataKind.METRIC.value: DataKind.CONSUMPTION}).sum().item()


class PhysicalMetricEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.PHYSICAL_METRICS
    _name = "physical_metric_evaluation"

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        results = kwargs.get("results", args[0])
        dataset = kwargs.get('dataset', args[1])
        for parameter, calculator in cls._parameter_calculators.items():
            dataset, results.loc[results.index[-1], parameter.to_abbrev_str()] = calculator.calculate(dataset, dataset)
        return dataset
