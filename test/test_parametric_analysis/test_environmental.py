import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import EnvironmentalMetric, PhysicalMetric, OtherParameters
from parameteric_evaluation.environmental import (
    EmissionSavingsRatio, TotalEmissions, BaselineEmissions, EmissionFactors
)


class TestEnvironmental(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

        # Mock emission factors
        self.mock_emission_factors = {
            'grid': 0.5,
            'inj': 0.1,
            'prod': 0.05,
            'bess': 100.0
        }

    @patch('parameteric_evaluation.environmental.read_csv')
    @patch('parameteric_evaluation.environmental.config')
    def test_emission_factors_read(self, mock_config, mock_read_csv):
        """Test EmissionFactors read method"""
        mock_config.get.return_value = "dummy_file.csv"
        mock_df = pd.DataFrame({'value': self.mock_emission_factors})
        mock_read_csv.return_value = mock_df

        factors = EmissionFactors.read("test.csv")

        self.assertIsInstance(factors, dict)

    @patch('parameteric_evaluation.environmental.EmissionFactors')
    def test_baseline_emissions_calculation(self, mock_emission_class):
        """Test BaselineEmissions calculation"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.return_value = 0.5
        mock_emission_class.return_value = mock_instance

        total_consumption = np.array([10.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.TOTAL_CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([total_consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        years = 20
        result, emissions = BaselineEmissions.calculate(input_da, years=years)

        # emissions = total_consumption * grid_factor * years
        expected = 10.0 * 24 * 0.5 * years
        self.assertAlmostEqual(emissions, expected, places=5)

    @patch('parameteric_evaluation.environmental.EmissionFactors')
    def test_total_emissions_calculation(self, mock_emission_class):
        """Test TotalEmissions calculation"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: self.mock_emission_factors.get(x, 0)
        mock_emission_class.return_value = mock_instance

        shared = np.array([5.0] * 24)
        withdrawn = np.array([10.0] * 24)
        production = np.array([8.0] * 24)
        injected = np.array([3.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [
                PhysicalMetric.SHARED_ENERGY,
                OtherParameters.WITHDRAWN_ENERGY,
                DataKind.PRODUCTION,
                OtherParameters.INJECTED_ENERGY
            ]
        }
        input_da = OmnesDataArray(
            data=np.array([shared, withdrawn, production, injected]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        years = 20
        battery_size = 10.0

        result, emissions = TotalEmissions.calculate(
            input_da,
            years=years,
            battery_size=battery_size
        )

        # emissions can be an OmnesDataArray, extract the value
        if hasattr(emissions, 'values'):
            emissions_value = float(emissions.values)
        else:
            emissions_value = emissions
        self.assertIsInstance(emissions_value, (int, float))

    @patch('parameteric_evaluation.environmental.EmissionFactors')
    def test_emission_savings_ratio_calculation(self, mock_emission_class):
        """Test EmissionSavingsRatio calculation"""
        mock_instance = MagicMock()
        mock_emission_class.return_value = mock_instance

        # Create results with baseline and total emissions
        battery_size = 10.0
        number_of_families = 50

        coords = {
            DataKind.METRIC.value: [
                EnvironmentalMetric.BASELINE_EMISSIONS,
                EnvironmentalMetric.TOTAL_EMISSIONS
            ],
            DataKind.BATTERY_SIZE.value: [battery_size],
            DataKind.NUMBER_OF_FAMILIES.value: [number_of_families]
        }

        baseline = 1000.0
        total = 600.0

        results = OmnesDataArray(
            data=np.array([[[baseline]], [[total]]]),
            dims=[DataKind.METRIC.value, DataKind.BATTERY_SIZE.value, DataKind.NUMBER_OF_FAMILIES.value],
            coords=coords
        )

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        result, esr = EmissionSavingsRatio.calculate(
            input_da,
            results_of_previous_calculations=results,
            battery_size=battery_size,
            number_of_families=number_of_families
        )

        # ESR = (baseline - total) / baseline = (1000 - 600) / 1000 = 0.4
        expected_esr = (baseline - total) / baseline
        np.testing.assert_almost_equal(esr, expected_esr, decimal=5)

    @patch('parameteric_evaluation.environmental.EmissionFactors')
    def test_emission_savings_ratio_with_zero_baseline(self, mock_emission_class):
        """Test EmissionSavingsRatio with zero baseline"""
        mock_instance = MagicMock()
        mock_emission_class.return_value = mock_instance

        battery_size = 10.0
        number_of_families = 50

        coords = {
            DataKind.METRIC.value: [
                EnvironmentalMetric.BASELINE_EMISSIONS,
                EnvironmentalMetric.TOTAL_EMISSIONS
            ],
            DataKind.BATTERY_SIZE.value: [battery_size],
            DataKind.NUMBER_OF_FAMILIES.value: [number_of_families]
        }

        baseline = 0.0
        total = 0.0

        results = OmnesDataArray(
            data=np.array([[[baseline]], [[total]]]),
            dims=[DataKind.METRIC.value, DataKind.BATTERY_SIZE.value, DataKind.NUMBER_OF_FAMILIES.value],
            coords=coords
        )

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        result, esr = EmissionSavingsRatio.calculate(
            input_da,
            results_of_previous_calculations=results,
            battery_size=battery_size,
            number_of_families=number_of_families
        )

        # With zero baseline, should return the baseline value
        np.testing.assert_almost_equal(esr, baseline, decimal=5)

    def test_environmental_calculator_keys(self):
        """Test environmental calculator keys"""
        self.assertEqual(EmissionSavingsRatio._key, EnvironmentalMetric.ESR)
        self.assertEqual(TotalEmissions._key, EnvironmentalMetric.TOTAL_EMISSIONS)
        self.assertEqual(BaselineEmissions._key, EnvironmentalMetric.BASELINE_EMISSIONS)

    @patch('parameteric_evaluation.environmental.EmissionFactors')
    def test_baseline_emissions_with_zero_years(self, mock_emission_class):
        """Test BaselineEmissions with zero years"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.return_value = 0.5
        mock_emission_class.return_value = mock_instance

        total_consumption = np.array([10.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.TOTAL_CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([total_consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, emissions = BaselineEmissions.calculate(input_da, years=0)

        self.assertEqual(emissions, 0.0)


if __name__ == '__main__':
    unittest.main()
