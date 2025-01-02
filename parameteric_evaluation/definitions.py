from enum import auto

from utility.definitions import OrderedEnum


class ParametricEvaluationType(OrderedEnum):
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    SELF_CONSUMPTION_FOR_TIME_AGGREGATION = \
        "self_consumption_for_time_aggregations"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    ALL = "all"


class Parameter(OrderedEnum):
    SHARED_ENERGY = "Shared energy"
    SELF_CONSUMPTION = "Self consumption"
    SELF_SUFFICIENCY = "Self sufficiency"
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = auto()
