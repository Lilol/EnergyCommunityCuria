from typing import Iterable

from data_storage.omnes_data_array import OmnesDataArray
from operation.definitions import Status


class Operation:
    _name = "operation"

    def __init__(self, name=_name, *args, **kwargs):
        self.name = name
        self.status = Status.INVALID
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        raise NotImplementedError


