from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from output.write import Write
from parameteric_evaluation.battery import Battery
from parameteric_evaluation.dimensions import power_to_energy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import TotalConsumption


class MetricEvaluator:
    _parametric_evaluator = ParametricEvaluator.create()

    @classmethod
    def calculate_metrics(cls, parameters):
        # Get plants sizes and number of users
        ds = DataStore()
        data_plants = ds["data_plants"]
        n_users = len(data_plants.user)
        pv_sizes = data_plants.sel({DataKind.USER_DATA.value: DataKind.POWER}).values.flatten()
        if len(pv_sizes) < len(data_plants):
            raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")

        # Initialize results
        results = OmnesDataArray(
            dims=[DataKind.NUMBER_OF_FAMILIES.value, DataKind.BATTERY_SIZE.value, DataKind.METRIC.value],
            coords={DataKind.NUMBER_OF_FAMILIES.value: parameters.number_of_families,
                    DataKind.BATTERY_SIZE.value: parameters.bess_sizes, DataKind.METRIC.value: []})
        energy_year = ds["energy_year"]
        p_prod = energy_year.sel({DataKind.CALCULATED.value: DataKind.PRODUCTION})
        e_prod = power_to_energy(p_prod)

        # Evaluate each scenario
        for i, (n_fam, bess_size) in enumerate(parameters.combinations):
            # Calculate withdrawn power
            ds["energy_year"], p_with = TotalConsumption.calculate(energy_year, num_families=n_fam)
            e_cons = power_to_energy(p_with)

            # Manage BESS, if present
            p_inj = p_prod - Battery(bess_size).manage_bess(p_prod, p_with)

            for name, evaluator in cls._parametric_evaluator.items():
                evaluator.invoke(results, p_inj=p_inj, p_with=p_with, e_cons=e_cons, e_prod=e_prod, pv_sizes=pv_sizes,
                                 bess_size=bess_size, n_users=n_users + n_fam)

        Write().execute(results, name="results")
