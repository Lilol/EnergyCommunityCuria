from enum import auto

from input.definitions import DataKind
from utility.definitions import OrderedEnum


class Parameter(OrderedEnum):
    def to_abbrev_str(self):
        abbrev_dictionary = self._get_abbrev_mapping()
        return abbrev_dictionary.get(self, None)

    @classmethod
    def _get_abbrev_mapping(cls):
        raise NotImplementedError("Subclasses must implement _get_abbrev_mapping")


class PhysicalMetric(Parameter):
    SHARED_ENERGY = "Shared energy"
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    INVALID = auto()

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SHARED_ENERGY: "e_sh", cls.INJECTED_ENERGY: "e_inj", cls.WITHDRAWN_ENERGY: "e_with", }


class EnvironmentalMetric(Parameter):
    ESR = "Emissions savings ratio"
    TOTAL_EMISSIONS = "Total emissions"
    BASELINE_EMISSIONS = "Baseline emissions"
    INVALID = auto()

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.ESR: "esr", cls.TOTAL_EMISSIONS: "em_tot", cls.BASELINE_EMISSIONS: "e_base", }


class EconomicMetric(Parameter):
    CAPEX = "Capex"
    CAPEX_PV = "Capex PV"
    OPEX = "Opex"
    INVALID = auto()

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.CAPEX: "capex", cls.OPEX: "opex", cls.CAPEX_PV: "capex_pv", }


class LoadMatchingMetric(Parameter):
    SELF_CONSUMPTION = "self_consumption"
    SELF_SUFFICIENCY = "self_sufficiency"
    INVALID = auto()

    @classmethod
    def _get_abbrev_mapping(cls):
        return {cls.SELF_CONSUMPTION: "sc", cls.SELF_SUFFICIENCY: "ss", }


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


def get_eval_metrics(evaluation_type):
    return {ParametricEvaluationType.PHYSICAL_METRICS: PhysicalMetric,
            ParametricEvaluationType.LOAD_MATCHING_METRICS: LoadMatchingMetric,
            ParametricEvaluationType.ECONOMIC_METRICS: EconomicMetric,
            ParametricEvaluationType.ENVIRONMENTAL_METRICS: EnvironmentalMetric, }.get(evaluation_type, None)


def calculate_shared_energy(data, n_fam):
    calc_sum_consumption(data, n_fam)
    data[DataKind.SHARED] = data.sel([DataKind.PRODUCTION, DataKind.CONSUMPTION]).min(axis="rows")


def calc_sum_consumption(data, n_fam):
    data[DataKind.CONSUMPTION] = data.sel(DataKind.CONSUMPTION_OF_FAMILIES) * n_fam + data.sel(
        DataKind.CONSUMPTION_OF_USERS)


def calculate_sc(df):
    return df[DataKind.SHARED].sum() / df[DataKind.PRODUCTION].sum()
