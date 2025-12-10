import logging

from data_storage.data_store import DataStore
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.output.write import WriteDataArray
from parameteric_evaluation.battery import Battery
from parameteric_evaluation.definitions import PhysicalMetric
from parameteric_evaluation.other_calculators import WithdrawnEnergy, InjectedEnergy
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from parameteric_evaluation.physical import TotalConsumption
from utility.configuration import config
from utility.time_utils import to_hours
from visualization.processing_visualization import plot_metrics

logger = logging.getLogger(__name__)


class MetricEvaluator:
    _parametric_evaluators = ParametricEvaluator.create_evaluators_based_on_configuration()

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
        results = OmnesDataArray(0.0, dims=[DataKind.NUMBER_OF_FAMILIES.value, DataKind.BATTERY_SIZE.value,
                                            DataKind.METRIC.value, DataKind.MUNICIPALITY.value],
                                 coords={DataKind.NUMBER_OF_FAMILIES.value: parameters.number_of_families,
                                         DataKind.BATTERY_SIZE.value: parameters.bess_sizes, DataKind.METRIC.value: [],
                                         DataKind.MUNICIPALITY.value: ds["energy_year"][
                                             DataKind.MUNICIPALITY.value].coords}, )

        # Evaluate each scenario
        for i, parameters in enumerate(parameters, 1):
            energy_year = ds["energy_year"].copy()
            n_fam = parameters[DataKind.NUMBER_OF_FAMILIES.value]
            bess_size = parameters[DataKind.BATTERY_SIZE.value]
            logger.info(
                f"Evaluating scenario no. {i} with number of families: {n_fam} and battery size: {bess_size} kWh")
            # Calculate withdrawn power
            energy_year, _ = TotalConsumption.calculate(energy_year, number_of_families=n_fam)
            logger.info(
                f"Total annual energy consumption: {energy_year.sel({DataKind.CALCULATED.value: PhysicalMetric.TOTAL_CONSUMPTION}).sum():.0f} kWh")
            energy_year, _ = WithdrawnEnergy.calculate(energy_year)
            energy_year, _ = InjectedEnergy.calculate(energy_year)
            # Manage BESS, if present
            energy_year = Battery(bess_size, t_hours=to_hours(config.get("time", "resolution"))).manage_bess(
                energy_year)

            for name, evaluator in cls._parametric_evaluators.items():
                if name.value == "invalid":
                    continue
                energy_year, results = evaluator.invoke(energy_year, results, parameters, pv_sizes=pv_sizes,
                                                        battery_size=bess_size, number_of_families=n_fam,
                                                        number_of_users=n_users)

            plot_metrics(results, n_fam=n_fam, bess_size=bess_size)
            WriteDataArray().execute(results, filename=f"results_n_fam_{n_fam}_bess_size_{bess_size}.csv")
