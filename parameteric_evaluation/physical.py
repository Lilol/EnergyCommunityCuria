from abc import abstractmethod
from typing import Iterable

import xarray as xr

from data_storage.dataset import OmnesDataArray
from io_operation.input import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class PhysicalParameterCalculator(Calculator):
    _key = PhysicalMetric.INVALID

    @classmethod
    @abstractmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        pass


class SharedEnergy(PhysicalParameterCalculator):
    _key = PhysicalMetric.SHARED_ENERGY

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        dx = input_da.sel({DataKind.CALCULATED.value: [PhysicalMetric.INJECTED_ENERGY,
                                                       PhysicalMetric.WITHDRAWN_ENERGY]}).min().assign_coords(
            {DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        input_da = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return input_da, input_da.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum()


class TotalConsumption(PhysicalParameterCalculator):
    _key = PhysicalMetric.TOTAL_CONSUMPTION

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        num_families = kwargs.get('num_families')
        dx = (input_da.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_FAMILIES}) * num_families + input_da.sel(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
            {DataKind.CALCULATED.value: DataKind.CONSUMPTION})
        output = xr.concat([input_da, dx], dim=DataKind.CALCULATED.value)
        return output, output.sel({DataKind.CALCULATED.value: DataKind.CONSUMPTION}).sum().item()


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
