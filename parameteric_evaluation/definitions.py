from utility.definitions import OrderedEnum


class Parameter(OrderedEnum):
    def to_abbrev_str(self):
        abbrev_dictionary = self._get_abbrev_mapping()
        return abbrev_dictionary.get(self, None)

    @classmethod
    def _get_abbrev_mapping(cls):
        raise NotImplementedError("Subclasses must implement _get_abbrev_mapping")

    @classmethod
    def get_all(cls):
        return cls.__members__.values()

    def valid(self):
        return self.value != "invalid"


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    TOTAL_CONSUMPTION = "Total consumption"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SHARED_ENERGY: "e_sh", cls.TOTAL_CONSUMPTION: "c_tot"}


class OtherParameters(Parameter):
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.INJECTED_ENERGY: "e_inj", cls.WITHDRAWN_ENERGY: "e_with"}


class BatteryPowerFlows(Parameter):
    STORED_ENERGY = "Stored energy"
    POWER_CHARGE = "Charging power"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.STORED_ENERGY: "e_stor", cls.POWER_CHARGE: "p_charge"}


class EnvironmentalMetric(Parameter):
    BASELINE_EMISSIONS = "Baseline emissions"
    TOTAL_EMISSIONS = "Total emissions"
    ESR = "Emissions savings ratio"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.ESR: "esr", cls.TOTAL_EMISSIONS: "em_tot", cls.BASELINE_EMISSIONS: "e_base", }


class EconomicMetric(Parameter):
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.CAPEX: "capex", cls.OPEX: "opex"}


class LoadMatchingMetric(Parameter):
    SELF_CONSUMPTION = "Self consumption"
    SELF_SUFFICIENCY = "Self sufficiency"
    SELF_PRODUCTION = "Self production"
    GRID_LIABILITY = "Grid liability"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SELF_CONSUMPTION: "sc", cls.SELF_SUFFICIENCY: "ss", }


class TimeAggregation(Parameter):
    THEORETICAL_LIMIT = "15min"
    HOUR = "hour"
    DAY = "dayofyear"
    MONTH = 'month'
    SEASON = 'season'
    YEAR = "year"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.HOUR: "hour", cls.MONTH: "month", cls.DAY: "month", cls.YEAR: "year",
                cls.THEORETICAL_LIMIT: "th_lim", }


class ParametricEvaluationType(OrderedEnum):
    DATASET_CREATION = "dataset_creation"
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    TIME_AGGREGATION = "time_aggregation"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    LOAD_MATCHING_METRICS = "load_matching"
    ALL = "all"
    INVALID = "invalid"


class CombinedMetric:
    def __init__(self, first, second):
        assert hasattr(first, "value") and hasattr(second, "value") and hasattr(first, "name") and hasattr(second,
                                                                                                           "name")
        self.first = first
        self.second = second

    @property
    def value(self):
        return self.first.value, self.second.value

    @property
    def name(self):
        return f"{self.first.name}, {self.second.name}"

    def valid(self):
        return self.first.valid() and self.second.valid()

    def __eq__(self, other):
        if not isinstance(other, CombinedMetric):
            return False
        return (self.first, self.second) == (other.first, other.second)

    def __hash__(self):
        return hash((self.first.value, self.second.value))

    def to_str(self):
        return f"{self.first.value}_{self.second.value}"

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def to_tuple(self):
        return (self.first.value, self.second.value)
