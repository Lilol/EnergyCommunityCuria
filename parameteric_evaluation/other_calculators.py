from typing import Iterable

from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import OtherParameters


class Equality(Calculator):
    _key = OtherParameters.INVALID
    _equate_to = None

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        new_coords = input_da.coords[DataKind.METRIC.value].data
        new_coords[new_coords == cls._equate_to] = cls._key
        input_da = input_da.assign_coords({DataKind.METRIC.value: (DataKind.METRIC.value, new_coords)})
        return input_da, input_da.sel({DataKind.METRIC.value: cls._key}).sum().item()


class InjectedEnergy(Equality):
    _key = OtherParameters.INJECTED_ENERGY
    _equate_to = DataKind.PRODUCTION


class WithdrawnEnergy(Equality):
    _key = OtherParameters.WITHDRAWN_ENERGY
    _equate_to = DataKind.CONSUMPTION
