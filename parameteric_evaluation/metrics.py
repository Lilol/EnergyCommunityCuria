import numpy as np
from pandas import DataFrame

from input.definitions import DataKind
from output.write import Write
from parameteric_evaluation.definitions import ParametricEvaluationType
from parameteric_evaluation.parametric_evaluation import ParametricEvaluator
from utility.dimensions import power_to_energy


def manage_bess(p_prod, p_cons, bess_size, t_min=None):
    """Manage BESS power flows to increase shared energy."""
    # Initialize BESS power array and stored energy
    p_bess = np.zeros_like(p_prod)
    if bess_size == 0:
        return p_bess

    # Get inputs
    p_bess_max = np.inf if t_min is None else bess_size / t_min

    e_stored = 0

    # Manage flows in all time steps
    for h, (p, c) in enumerate(zip(p_prod, p_cons)):
        # Power to charge the BESS (discharge if negative)
        b = p - c
        # Correct according to technical/physical limits
        if b < 0:
            b = max(b, -e_stored, -p_bess_max)
        else:
            b = min(b, bess_size - e_stored, p_bess_max)
        # Update BESS power array and stored energy
        p_bess[h] = b
        e_stored = e_stored + b

    return p_bess


class MetricEvaluator(ParametricEvaluator):
    _type = ParametricEvaluationType.METRICS

    @classmethod
    def define_output_df(cls, evaluation_types, scenarios):
        physical_metrics = ["sc", "ss", "e_sh", "e_inj",
                            "e_with"] if ParametricEvaluationType.PHYSICAL_METRICS in evaluation_types else []
        environmental_metrics = ["esr", "em_tot",
                                 "em_base"] if ParametricEvaluationType.ENVIRONMENTAL_METRICS in evaluation_types else []
        economic_metrics = ["capex", ] if ParametricEvaluationType.ECONOMIC_METRICS in evaluation_types else []
        return DataFrame(index=scenarios.index, columns=physical_metrics + environmental_metrics + economic_metrics)

    @classmethod
    def calculate_metrics(cls):
        scenarios = DataFrame(data=parameters.combinations,
                              columns=[DataKind.NUMBER_OF_FAMILIES, DataKind.BATTERY_SIZE])
        # Get plants sizes and number of users
        n_users = len(ds["data_users"])
        data_plants = ds["data_plants"]
        pv_sizes = list(data_plants.loc[data_plants[DataKind.USER_TYPE] == 'pv', DataKind.POWER])
        if len(pv_sizes) < len(data_plants):
            raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")
        # Initialize results
        results = cls.define_output_df(evaluation_types, scenarios)
        p_prod = energy_year.sel(user=DataKind.PRODUCTION)
        e_prod = power_to_energy(p_prod)
        p_cons = energy_year.sel(user=DataKind.CONSUMPTION_OF_USERS)
        p_fam = energy_year.sel(user=DataKind.CONSUMPTION_OF_FAMILIES)
        # Evaluate each scenario
        for i, scenario in scenarios.iterrows():
            # Get configuration
            n_fam = scenario[DataKind.NUMBER_OF_FAMILIES]
            bess_size = scenario[DataKind.BATTERY_SIZE]

            # Calculate withdrawn power
            p_with = p_cons + n_fam * p_fam
            # Manage BESS, if present
            p_inj = p_prod - manage_bess(p_prod, p_with, bess_size)

            # Eval REC
            e_cons = power_to_energy(p_with)
            if ParametricEvaluationType.PHYSICAL_METRICS in evaluation_types:
                results.loc[i, ["sc", "ss", "e_sh", "e_inj", "e_with"]] = eval_physical_parameters(p_inj, p_with)

            # Evaluate emissions
            if ParametricEvaluationType.ENVIRONMENTAL_METRICS in evaluation_types:
                results.loc[i, ["esr", "em_tot", "em_base"]] = eval_co2(results.loc[i, "e_sh"],
                                                                        e_cons=results.loc[i, "e_with"],
                                                                        e_inj=results.loc[i, "e_inj"], e_prod=e_prod)

            # Evaluate CAPEX
            if ParametricEvaluationType.ECONOMIC_METRICS in evaluation_types:
                results.loc[i, "capex"] = eval_capex(pv_sizes, bess_size=bess_size, n_users=n_users + n_fam)
        Write().write(results, "results")
