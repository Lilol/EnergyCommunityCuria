from enum import auto

from utility.definitions import OrderedEnum


class Parameter(OrderedEnum):
    def to_abbrev_str(self):
        pass


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    INVALID = auto()

    def to_abbrev_str(self):
        abbrev_dictionary = {PhysicalMetric.SHARED_ENERGY: "e_sh", PhysicalMetric.INJECTED_ENERGY: "e_inj",
                             PhysicalMetric.WITHDRAWN_ENERGY: "e_with"}
        return abbrev_dictionary[self]


class EnvironmentalMetric(Parameter):
    ESR = "Emissions savings ratio"
    TOTAL_EMISSIONS = "Total emissions"
    BASELINE_EMISSIONS = "Baseline emissions"
    INVALID = auto()

    def to_abbrev_str(self):
        abbrev_dictionary = {EnvironmentalMetric.ESR: "esr", EnvironmentalMetric.TOTAL_EMISSIONS: "em_tot",
                             EnvironmentalMetric.BASELINE_EMISSIONS: "e_base"}
        return abbrev_dictionary[self]


class EconomicMetric(Parameter):
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = auto()

    def to_abbrev_str(self):
        abbrev_dictionary = {EconomicMetric.CAPEX: "capex", EconomicMetric.OPEX: "opex"}
        return abbrev_dictionary[self]


class LoadMatchingMetric(Parameter):
    SELF_CONSUMPTION = "self_consumption"
    SELF_SUFFICIENCY = "self_sufficiency"
    INVALID = auto()

    def to_abbrev_str(self):
        abbrev_dictionary = {LoadMatchingMetric.SELF_CONSUMPTION: "sc", LoadMatchingMetric.SELF_SUFFICIENCY: "ss"}
        return abbrev_dictionary[self]


class ParametricEvaluationType(OrderedEnum):
    DATASET_CREATION = "dataset_creation"
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    SELF_CONSUMPTION_FOR_TIME_AGGREGATIONS = "self_consumption_for_time_aggregations"
    TIME_AGGREGATION = "time_aggregations"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    LOAD_MATCHING_METRICS = "load_matching"
    ALL = "all"
    INVALID = "invalid"


evaluation_type = {ParametricEvaluationType.PHYSICAL_METRICS: PhysicalMetric,
                   ParametricEvaluationType.LOAD_MATCHING_METRICS: LoadMatchingMetric,
                   ParametricEvaluationType.ECONOMIC_METRICS: EconomicMetric,
                   ParametricEvaluationType.ENVIRONMENTAL_METRICS: EnvironmentalMetric,}
