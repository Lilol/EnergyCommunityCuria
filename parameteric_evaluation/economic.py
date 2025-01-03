from typing import Iterable

from data_storage.dataset import OmnesDataArray
from parameteric_evaluation import MetricEvaluator
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, Parameter


class CapexPv(Calculator):
    _parameter_calculated = Parameter.CAPEX

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        """Evaluate investment cost (CAPEX) of a PV system depending on the size."""
        pv_size = kwargs.pop('pv_size', args[0])
        # if pv_size < 10 :
        #     capex_pv = 1900
        # elif pv_size < 35 :
        #     capex_pv = -6 * pv_size + 1960
        # elif pv_size < 125 :
        #     capex_pv = -7.2 * pv_size + 2002.8
        # elif pv_size < 600 :
        #     capex_pv = -0.74 * pv_size + 1192.1
        # else :
        #     capex_pv=750
        if pv_size < 20:
            c_pv = 1500
        elif pv_size < 200:
            c_pv = 1200
        elif pv_size < 600:
            c_pv = 1100
        else:
            c_pv = 1050
        return c_pv * pv_size


class Capex(Calculator):
    _parameter_calculated = Parameter.CAPEX
    c_bess = 300
    c_user = 100

    @classmethod
    def calculate(cls, input_da: OmnesDataArray, output: OmnesDataArray | None, *args,
                  **kwargs) -> None | OmnesDataArray | float | Iterable[OmnesDataArray]:
        """Evaluate CAPEX of a REC, given PV sizes, BESS size(s) and number of users."""

        # Initialize CAPEX
        capex = 0

        # Add cost of PVS
        for pv_size in kwargs.pop('pv_sizes', args[0]):
            capex += CapexPv.calculate(pv_size, None)

        # Add cost of BESS
        capex += kwargs.pop('bess_size', args[1]) * cls.c_bess

        # Add cost of users
        capex += kwargs.pop('n_users', args[2]) * cls.c_user
        return capex


class EconomicEvaluator(MetricEvaluator):
    _type = ParametricEvaluationType.ECONOMIC_METRICS

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float:
        return Capex.calculate(*args, **kwargs)
