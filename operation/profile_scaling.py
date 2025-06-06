from typing import Iterable

from data_storage.omnes_data_array import OmnesDataArray
from operation.definitions import ScalingMethod
from operation.operation import Operation
from utility import configuration
from utility.subclass_registration_base import SubclassRegistrationBase


class ScaleProfile(Operation, SubclassRegistrationBase):
    _name = "profile_scaler"
    _key = ScalingMethod.INVALID

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, *operands: Iterable[OmnesDataArray], **kwargs) -> OmnesDataArray:
        raise NotImplementedError

    @classmethod
    def create(cls, name=_name, *args, **kwargs):
        return cls._subclasses.get(configuration.config.get("profile", "scaling_method"), None)(name, *args, **kwargs)
