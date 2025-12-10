import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import EconomicMetric
from parameteric_evaluation.economic import Capex, Opex, CostOfEquipment


class TestEconomic(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

        # Mock cost data
        self.mock_cost_data = pd.DataFrame({
            'cost': [1000.0, 1200.0, 800.0, 50.0, 60.0, 40.0, 500.0, 100.0]
        }, index=pd.MultiIndex.from_tuples([
            ('pv', 'capex', 10.0),
            ('pv', 'capex', 50.0),
            ('pv', 'capex', 100.0),
            ('pv', 'opex', 10.0),
            ('pv', 'opex', 50.0),
            ('pv', 'opex', 100.0),
            ('bess', 'capex', np.nan),
            ('bess', 'opex', np.nan)
        ], names=['equipment', 'cost_type', 'max_size']))

    @patch('parameteric_evaluation.economic.read_csv')
    @patch('parameteric_evaluation.economic.config')
    def test_cost_of_equipment_read(self, mock_config, mock_read_csv):
        """Test CostOfEquipment read method"""
        mock_config.get.return_value = "dummy_file.csv"
        mock_read_csv.return_value = self.mock_cost_data

        costs = CostOfEquipment.read("test.csv")

        self.assertIsInstance(costs, pd.DataFrame)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_capex_of_pv_small_size(self, mock_cost_class):
        """Test CAPEX calculation for small PV system"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.return_value = 1000.0
        mock_cost_class.return_value = mock_instance

        pv_size = 5.0
        capex = Capex.capex_of_pv(pv_size)

        self.assertEqual(capex, 5000.0)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_capex_calculation(self, mock_cost_class):
        """Test total CAPEX calculation"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: {
            ("pv", "capex", 10.0): 1000.0,
            ("bess", "capex"): 500.0,
            ("user", "capex"): 100.0
        }.get(x if isinstance(x, tuple) else tuple(x))
        mock_cost_class.return_value = mock_instance

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        pv_sizes = [10.0, 15.0]
        battery_size = 20.0
        number_of_families = 50

        result, capex = Capex.calculate(
            input_da=input_da,
            pv_sizes=pv_sizes,
            battery_size=battery_size,
            number_of_families=number_of_families
        )

        self.assertIsInstance(capex, (int, float))
        self.assertGreater(capex, 0)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_capex_with_zero_battery(self, mock_cost_class):
        """Test CAPEX with zero battery size"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: {
            ("pv", "capex", 10.0): 1000.0,
            ("bess", "capex"): 500.0,
            ("user", "capex"): 100.0
        }.get(x if isinstance(x, tuple) else tuple(x))
        mock_cost_class.return_value = mock_instance

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        result, capex = Capex.calculate(
            input_da=input_da,
            pv_sizes=[10.0],
            battery_size=0.0,
            number_of_families=10
        )

        self.assertIsNotNone(capex)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_opex_calculation(self, mock_cost_class):
        """Test total OPEX calculation"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: {
            ("pv", "opex"): 50.0,
            ("bess", "opex"): 100.0
        }.get(x if isinstance(x, tuple) else tuple(x))
        mock_cost_class.return_value = mock_instance

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        pv_sizes = [10.0, 20.0]
        battery_size = 15.0

        result, opex = Opex.calculate(
            input_da=input_da,
            pv_sizes=pv_sizes,
            battery_size=battery_size
        )

        self.assertIsInstance(opex, (int, float))
        self.assertGreater(opex, 0)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_opex_with_multiple_pv(self, mock_cost_class):
        """Test OPEX with multiple PV systems"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: {
            ("pv", "opex"): 50.0,
            ("bess", "opex"): 100.0
        }.get(x if isinstance(x, tuple) else tuple(x))
        mock_cost_class.return_value = mock_instance

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        pv_sizes = [5.0, 10.0, 15.0]
        battery_size = 10.0

        result, opex = Opex.calculate(
            input_da=input_da,
            pv_sizes=pv_sizes,
            battery_size=battery_size
        )

        self.assertIsNotNone(opex)

    def test_capex_key(self):
        """Test Capex calculator key"""
        self.assertEqual(Capex._key, EconomicMetric.CAPEX)

    def test_opex_key(self):
        """Test Opex calculator key"""
        self.assertEqual(Opex._key, EconomicMetric.OPEX)

    @patch('parameteric_evaluation.economic.CostOfEquipment')
    def test_capex_with_empty_pv_list(self, mock_cost_class):
        """Test CAPEX with empty PV list"""
        mock_instance = MagicMock()
        mock_instance.__getitem__.side_effect = lambda x: {
            ("bess", "capex"): 500.0,
            ("user", "capex"): 100.0
        }.get(x if isinstance(x, tuple) else tuple(x))
        mock_cost_class.return_value = mock_instance

        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords={
                DataKind.TIME.value: self.time,
                DataKind.CALCULATED.value: ['test']
            }
        )

        result, capex = Capex.calculate(
            input_da=input_da,
            pv_sizes=[],
            battery_size=10.0,
            number_of_families=5
        )

        self.assertIsNotNone(capex)


if __name__ == '__main__':
    unittest.main()

