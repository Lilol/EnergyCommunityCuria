import numpy as np

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import OtherParameters, PhysicalMetric


class Equality(Calculator):
    _key = OtherParameters.INVALID
    _equate_to = None

    @classmethod
    def calculate(cls, input_da: OmnesDataArray | None = None,
                  results_of_previous_calculations: OmnesDataArray | None = None, *args, **kwargs) -> tuple[
        OmnesDataArray, float | None]:
        if input_da is None:
            return input_da, results_of_previous_calculations

        new_coords = input_da.coords[DataKind.CALCULATED.value].data
        # Use numpy array operations to check membership
        equate_to_present = np.any(new_coords == cls._equate_to)
        key_present = np.any(new_coords == cls._key)

        if not equate_to_present or key_present:
            return input_da, results_of_previous_calculations

        new_coords = new_coords.copy()
        new_coords[new_coords == cls._equate_to] = cls._key
        input_da = input_da.assign_coords({DataKind.CALCULATED.value: (DataKind.CALCULATED.value, new_coords)})
        return input_da, results_of_previous_calculations


class InjectedEnergy(Equality):
    _key = OtherParameters.INJECTED_ENERGY
    _equate_to = DataKind.PRODUCTION


class WithdrawnEnergy(Equality):
    _key = OtherParameters.WITHDRAWN_ENERGY
    _equate_to = PhysicalMetric.TOTAL_CONSUMPTION
