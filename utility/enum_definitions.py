from enum import Enum

from io_operation.input.definitions import DataKind, UserType, BillType, PvDataSource
from operation.definitions import Status, ScalingMethod


def convert_enum_to_value(x):
    return x.value if isinstance(x, Enum) else x


def convert_value_to_enum(x):
    # Bit hackish, but no better idea
    enums_to_try = (DataKind, UserType, BillType, PvDataSource, Status, ScalingMethod)
    y = x
    try:
        return float(x)
    except ValueError:
        for e in enums_to_try:
            try:
                y = e(x)
            except ValueError:
                try:
                    y = e[x]
                except KeyError:
                    pass
    return y
