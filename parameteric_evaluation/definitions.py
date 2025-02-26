from enum import auto

from utility.definitions import OrderedEnum


class ParametricEvaluationType(OrderedEnum):
    DATASET_CREATION = "dataset_creation"
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    SELF_CONSUMPTION_FOR_TIME_AGGREGATIONS ="self_consumption_for_time_aggregations"
    TIME_AGGREGATION = "time_aggregations"
    METRICS = "metrics"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    ALL = "all"
    INVALID = "invalid"


class Parameter(OrderedEnum):
    # Parameter -> Metric, Parameter!!!
    SHARED_ENERGY = "Shared energy"
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    ESR = "Emissions savings ratio"
    TOTAL_EMISSIONS = "Total emissions"
    BASELINE_EMISSIONS = "Baseline emissions"
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = auto()


class Metric(Parameter):
    SC = "self_consumption"
    SS = "self_sufficiency"
