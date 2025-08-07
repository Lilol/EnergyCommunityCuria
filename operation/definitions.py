from enum import auto

from utility.definitions import OrderedEnum


class Status(OrderedEnum):
    OK = 'ok'
    OPTIMAL = 'optimal'
    UNPHYSICAL = 'unphysical'
    UNFEASIBLE = 'unfeasible'
    UNKNOWN = 'unknown'
    ERROR = 'error'
    INVALID = 'invalid'


class ScalingMethod(OrderedEnum):
    IN_PROPORTION = "proportional"
    FLAT = "flat"
    TIME_OF_USE = "time_of_use"
    LINEAR_EQUATION = "linear"
    QUADRATIC_OPTIMIZATION = "quadratic_optimization"
    INVALID = "invalid"
