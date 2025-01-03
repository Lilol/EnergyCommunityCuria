from parameteric_evaluation import MetricEvaluator
from parameteric_evaluation.definitions import ParametricEvaluationType


class EnvironmentalEvaluator(MetricEvaluator):
    _type = ParametricEvaluationType.ENVIRONMENTAL_METRICS

    @classmethod
    def eval_co2(cls, e_sh, e_cons, e_prod, e_with=None, e_inj=None, bess_size=0, eps_grid=0.263, eps_inj=0,
                 eps_prod=0.05, eps_bess=175, n=20):
        """
        Calculates the CO2 emissions based on the shared energy, consumed energy,
        produced energy, and emission factors.

        Parameters:
        - e_sh (float): Shared energy.
        - e_cons (float): Consumed energy.
        - e_prod (float): Produced energy.
        - bess_size (float): Size of Battery Energy Storage System (BESS) in kWh.
            Default is 0.
        - eps_grid (float): Emission factor for energy from the grid.
            Default is 0.263 kgCO2eq/kWh.
        - eps_inj (float): Emission factor for injected energy.
            Default is -0 kgCO2eq/kWh.
        - eps_prod (float): Emission factor for produced energy (LCA).
            Default is 0.05 kgCO2eq/kWh.
        - eps_bess (float): Emission factor for BESS capacity.
            Default is 175 kgCO2eq/kWh.
        - n (int): Number of years considered. Default is 20.

        Returns:
        - Tuple[float, float, float]: Emissions savings ratio, total emissions,
            and baseline emissions.
        """
        # Get values of injections and withdrawals
        e_inj = e_prod if e_inj is None else e_inj
        e_with = e_cons if e_with is None else e_with

        # Evaluate total emissions
        em_tot = ((e_with - e_sh) * eps_grid + (e_inj - e_sh) * eps_inj + eps_prod * e_prod) * n + bess_size * eps_bess

        # Evaluate total emissions in base case
        em_base = (e_cons * eps_grid) * n

        # Evaluate emissions savings ratio
        esr = (em_base - em_tot) / em_base

        return esr, em_tot, em_base
