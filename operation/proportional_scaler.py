from typing import Iterable

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from operation import ScaleProfile
from operation.definitions import ScalingMethod, Status


class ScaleInProportion(ScaleProfile):
    _name = "proportional_profile_scaler"
    _key = ScalingMethod.IN_PROPORTION

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.status = Status.OPTIMAL

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        total_consumption_by_time_slots = operands[0]
        reference_profile = operands[1]
        day_type_count = DataStore()["day_count"].sel({DataKind.MONTH.value: reference_profile.month.values})
        return (reference_profile / (day_type_count * reference_profile.sum(
            DataKind.HOUR.value)).sum() * total_consumption_by_time_slots.sum()).squeeze()
