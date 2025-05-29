import logging

from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.output.write import Write
from parameteric_evaluation.battery import Battery
from parameteric_evaluation.definitions import PhysicalMetric
from parameteric_evaluation.other_calculators import WithdrawnEnergy, InjectedEnergy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import TotalConsumption

logger = logging.getLogger(__name__)


class MetricEvaluator:
    _parametric_evaluator = ParametricEvaluator.create()

    @classmethod
    def calculate_metrics(cls, parameters):
        # Get plants sizes and number of users
        ds = DataStore()
        data_plants = ds["data_plants"]
        n_users = len(data_plants.user)
        pv_sizes = data_plants.sel({DataKind.USER_DATA.value: DataKind.POWER}).astype(float).values.flatten()
        if len(pv_sizes) < len(data_plants):
            raise Warning("Some plants are not PV, add CAPEX manually and comment this Warning.")

        # Initialize results dataarray
        energy_year = ds["energy_year"]
        results = OmnesDataArray(
            dims=[DataKind.NUMBER_OF_FAMILIES.value, DataKind.BATTERY_SIZE.value, DataKind.METRIC.value,
                  DataKind.MUNICIPALITY.value],
            coords={DataKind.NUMBER_OF_FAMILIES.value: parameters.number_of_families,
                    DataKind.BATTERY_SIZE.value: parameters.bess_sizes, DataKind.METRIC.value: [],
                    DataKind.MUNICIPALITY.value: energy_year[DataKind.MUNICIPALITY.value].coords}, )

        # Evaluate each scenario
        for i, (n_fam, bess_size) in enumerate(parameters.combinations, 1):
            logger.info(
                f"Evaluating scenario no. {i} with number of families: {n_fam} and battery size: {bess_size} kWh")
            # Calculate withdrawn power
            energy_year, results = TotalConsumption.calculate(energy_year, num_families=n_fam)
            logger.info(
                f"Total annual energy consumption: {energy_year.sel({DataKind.CALCULATED.value: PhysicalMetric.TOTAL_CONSUMPTION}).sum():.0f} kWh")
            energy_year, _ = WithdrawnEnergy.calculate(energy_year)
            energy_year, _ = InjectedEnergy.calculate(energy_year)
            # Manage BESS, if present
            Battery(bess_size).manage_bess(energy_year)

            for name, evaluator in cls._parametric_evaluator.items():
                energy_year, results = evaluator.invoke(energy_year, results, pv_sizes=pv_sizes, battery_size=bess_size,
                                           number_of_families=n_fam, number_of_users=n_users)

        Write().execute(results, filename="results")
