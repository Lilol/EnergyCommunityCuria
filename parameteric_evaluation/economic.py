from typing import Iterable

from numpy import nan
from pandas import read_csv, IndexSlice

from data_storage.dataset import OmnesDataArray
from io_operation.input import ParametersFromFile
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, EconomicMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from utility.configuration import config


class CostOfEquipment(ParametersFromFile):
    _filename = config.get("parametric_evaluation", "cost_configuration_file")

    @classmethod
    def read(cls, filename):
        return read_csv(filename, header=0, index_col=(0, 1, 2))

    @classmethod
    def __getitem__(cls, item):
        try:
            equipment, cost_type, size = item
        except ValueError:
            equipment, cost_type = item
            return cls._parameters.loc[cls._parameters.index == (equipment, cost_type, nan), "cost"]
        else:
            for (_, _, max_size), params in cls._parameters.loc[IndexSlice[equipment, cost_type, :], "cost"].iterrows():
                if size <= max_size:
                    return params["cost"]


class Capex(Calculator):
    _key = EconomicMetric.CAPEX

    @classmethod
    def capex_of_pv(cls, pv_size):
        """Evaluate investment cost (CAPEX) of a PV system depending on the size."""

        return CostOfEquipment()["pv", "capex", pv_size] * pv_size

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        """Evaluate CAPEX of a REC, given PV sizes, BESS size(s) and number of users."""

        # Add cost of PVS
        capex = sum(cls.capex_of_pv(pv_size) for pv_size in kwargs.get('pv_sizes', []))

        # Add cost of BESS
        capex += kwargs.get('bess_size') * CostOfEquipment()["bess", "capex"]

        # Add cost of users
        capex += kwargs.get('n_users') * CostOfEquipment()["user", "capex"]
        return capex


class Opex(Calculator):
    _key = EconomicMetric.OPEX

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None = None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray] | tuple[
        OmnesDataArray, float | None]:
        """Evaluate OPEX of a REC, given PV sizes and BESS size(s)."""
        # Add cost of PVS
        opex = sum(CostOfEquipment()["opex_pv"]*pv_size for pv_size in kwargs.get('pv_sizes', []))

        # Add cost of BESS
        opex += kwargs.get('bess_size') * CostOfEquipment()["bess", "opex"]

        return opex


class EconomicEvaluator(ParametricEvaluator):
    _key = ParametricEvaluationType.ECONOMIC_METRICS
    _name = "economic_evaluation"
