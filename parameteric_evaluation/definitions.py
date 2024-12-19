from utility.definitions import OrderedEnum


class ParametricEvaluationType(OrderedEnum):
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    ALL = "all"
