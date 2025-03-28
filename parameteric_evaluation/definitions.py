import xarray as xr

from input.definitions import DataKind
from utility.definitions import OrderedEnum


class Parameter(DataKind):
    def to_abbrev_str(self):
        abbrev_dictionary = self._get_abbrev_mapping()
        return abbrev_dictionary.get(self, None)

    @classmethod
    def _get_abbrev_mapping(cls):
        raise NotImplementedError("Subclasses must implement _get_abbrev_mapping")

    @classmethod
    def get_all(cls):
        return cls.__members__.values()


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    TOTAL_CONSUMPTION = "Total consumption"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SHARED_ENERGY: "e_sh", cls.INJECTED_ENERGY: "e_inj", cls.WITHDRAWN_ENERGY: "e_with", }


class EnvironmentalMetric(Parameter):
    ESR = "Emissions savings ratio"
    TOTAL_EMISSIONS = "Total emissions"
    BASELINE_EMISSIONS = "Baseline emissions"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.ESR: "esr", cls.TOTAL_EMISSIONS: "em_tot", cls.BASELINE_EMISSIONS: "e_base", }


class EconomicMetric(Parameter):
    CAPEX = "Capex"
    CAPEX_PV = "Capex PV"
    OPEX = "Opex"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.CAPEX: "capex", cls.OPEX: "opex", cls.CAPEX_PV: "capex_pv", }


class LoadMatchingMetric(Parameter):
    SELF_CONSUMPTION = "Self consumption"
    SELF_SUFFICIENCY = "Self sufficiency"
    INVALID = "invalid"

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SELF_CONSUMPTION: "sc", cls.SELF_SUFFICIENCY: "ss", }


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


def calculate_sc(data):
    return data.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY}).sum() / data.sel(
        {DataKind.CALCULATED.value: DataKind.PRODUCTION}).sum()
