import xarray as xr

from input.definitions import DataKind
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


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
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


def calculate_shared_energy(data):
    dx = data.sel({DataKind.USER.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]}).min().assign_coords(
        {DataKind.USER.value: DataKind.SHARED})
    data = xr.concat([data, dx], dim=DataKind.USER.value)
    return data


def calc_sum_consumption(data, n_fam):
    dx = (data.sel({DataKind.USER.value: DataKind.CONSUMPTION_OF_FAMILIES}) * n_fam + data.sel(
        {DataKind.USER.value: DataKind.CONSUMPTION_OF_USERS})).assign_coords(
        {DataKind.USER.value: DataKind.CONSUMPTION})
    data = xr.concat([data, dx], dim=DataKind.USER.value)
    return data


def calculate_sc(data):
    return data.sel({DataKind.USER.value: DataKind.SHARED}).sum() / data.sel(
        {DataKind.USER.value: DataKind.PRODUCTION}).sum()
