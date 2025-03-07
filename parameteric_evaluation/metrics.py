from pandas import DataFrame

from data_storage.data_store import DataStore
from input.definitions import DataKind
from output.write import Write
from parameteric_evaluation.battery import Battery
from parameteric_evaluation.definitions import calc_sum_consumption
from parameteric_evaluation.dimensions import power_to_energy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class MetricEvaluator:
    _parametric_evaluator = ParametricEvaluator.create()

    @classmethod
    def calculate_metrics(cls, parameters):
        scenarios = DataFrame(data=parameters.combinations,
                              columns=[DataKind.NUMBER_OF_FAMILIES, DataKind.BATTERY_SIZE])
        # Get plants sizes and number of users
        ds = DataStore()
        data_plants = ds["data_plants"]
        n_users = len(data_plants)
        energy_year = ds["energy_year"]
        pv_sizes = list(data_plants.loc[data_plants[DataKind.USER_TYPE] == 'pv', DataKind.POWER])
        if len(pv_sizes) < len(data_plants):
            raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")

        # Initialize results
        results = DataFrame(index=scenarios.index)
        p_prod = energy_year.sel(user=DataKind.PRODUCTION)
        e_prod = power_to_energy(p_prod)

        # Evaluate each scenario
        for i, scenario in scenarios.iterrows():
            # Get configuration
            n_fam = scenario[DataKind.NUMBER_OF_FAMILIES]
            bess_size = scenario[DataKind.BATTERY_SIZE]

            # Calculate withdrawn power
            p_with = calc_sum_consumption(energy_year, n_fam)

            # Manage BESS, if present
            p_inj = p_prod - Battery(bess_size).manage_bess(p_prod, p_with)

            # Eval REC
            e_cons = power_to_energy(p_with)

            cls._parametric_evaluator.invoke(results, p_inj=p_inj, p_with=p_with, e_sh=results.loc[i, "e_sh"],
                                             e_cons=e_cons, e_inj=results.loc[i, "e_inj"], e_prod=e_prod,
                                             pv_sizes=pv_sizes, bess_size=bess_size, n_users=n_users + n_fam)

        Write().write(results, "results")
