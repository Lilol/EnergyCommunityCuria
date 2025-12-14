import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import PhysicalMetric, OtherParameters, LoadMatchingMetric, TimeAggregation
from parameteric_evaluation.time_aggregation_evaluation import TimeAggregationParameterCalculator


class TestTimeAggregationEvaluation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_theoretical_limit(self, mock_phys_calc):
        """Test time aggregation with theoretical limit (no aggregation)"""
        # Mock the calculator creation and calculation
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.75)
        mock_phys_calc.create.return_value = mock_calc

        production = np.array([10.0] * 12 + [5.0] * 12)
        consumption = np.array([5.0] * 12 + [10.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # Test that theoretical limit doesn't aggregate
        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.THEORETICAL_LIMIT
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        result, value = TestCalc.calculate(input_da)

        self.assertIsNotNone(value)
        self.assertEqual(value, 0.75)

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_hourly(self, mock_phys_calc):
        """Test hourly time aggregation"""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.8)
        mock_phys_calc.create.return_value = mock_calc

        # Create data for multiple days
        time_extended = pd.date_range(start='2023-01-01', periods=48, freq='h')
        production = np.array([10.0] * 24 + [12.0] * 24)
        consumption = np.array([5.0] * 24 + [6.0] * 24)

        coords = {
            DataKind.TIME.value: time_extended,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.HOUR
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        result, value = TestCalc.calculate(input_da)

        self.assertIsInstance(value, (int, float))
        self.assertEqual(value, 0.8)

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_daily(self, mock_phys_calc):
        """Test daily time aggregation"""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.65)
        mock_phys_calc.create.return_value = mock_calc

        # Create data for multiple days
        time_extended = pd.date_range(start='2023-01-01', periods=72, freq='h')
        production = np.random.rand(72) * 10
        consumption = np.random.rand(72) * 8

        coords = {
            DataKind.TIME.value: time_extended,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.DAY
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        result, value = TestCalc.calculate(input_da)

        self.assertIsNotNone(value)
        self.assertEqual(value, 0.65)

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_monthly(self, mock_phys_calc):
        """Test monthly time aggregation"""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.70)
        mock_phys_calc.create.return_value = mock_calc

        # Create data for full year
        time_extended = pd.date_range(start='2023-01-01', periods=365*24, freq='h')
        production = np.random.rand(365*24) * 10
        consumption = np.random.rand(365*24) * 8

        coords = {
            DataKind.TIME.value: time_extended,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.MONTH
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        result, value = TestCalc.calculate(input_da)

        self.assertIsNotNone(value)
        self.assertEqual(value, 0.70)

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_yearly(self, mock_phys_calc):
        """Test yearly time aggregation"""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.68)
        mock_phys_calc.create.return_value = mock_calc

        time_extended = pd.date_range(start='2023-01-01', periods=365*24, freq='h')
        production = np.random.rand(365*24) * 10
        consumption = np.random.rand(365*24) * 8

        coords = {
            DataKind.TIME.value: time_extended,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.YEAR
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        result, value = TestCalc.calculate(input_da)

        self.assertIsNotNone(value)
        self.assertEqual(value, 0.68)

    def test_time_aggregation_calculator_key(self):
        """Test TimeAggregationParameterCalculator key attribute"""
        class TestCalc(TimeAggregationParameterCalculator):
            _key = TimeAggregation.HOUR
            _aggregation = TimeAggregation.HOUR
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        self.assertEqual(TestCalc._key, TimeAggregation.HOUR)

    @patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator')
    def test_time_aggregation_with_self_sufficiency(self, mock_phys_calc):
        """Test time aggregation with self sufficiency metric"""
        # Mock the calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = (MagicMock(), 0.85)
        mock_phys_calc.create.return_value = mock_calc

        production = np.array([10.0] * 24)
        consumption = np.array([8.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        class TestCalc(TimeAggregationParameterCalculator):
            _aggregation = TimeAggregation.THEORETICAL_LIMIT
            _metric = LoadMatchingMetric.SELF_SUFFICIENCY

        result, value = TestCalc.calculate(input_da)

        self.assertIsNotNone(value)
        self.assertEqual(value, 0.85)

    @patch('parameteric_evaluation.time_aggregation_evaluation.plot_shared_energy')
    @patch('parameteric_evaluation.time_aggregation_evaluation.WriteDataArray')
    def test_time_aggregation_evaluator_invoke(self, mock_write, mock_plot):
        """Test TimeAggregationEvaluator invoke method"""
        from parameteric_evaluation.time_aggregation_evaluation import TimeAggregationEvaluator

        # Create proper mock dataset with required structure
        time = pd.date_range(start='2023-01-01', periods=24, freq='h')
        production = np.array([10.0] * 24)
        consumption = np.array([8.0] * 24)

        coords = {
            DataKind.TIME.value: time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        mock_dataset = OmnesDataArray(
            data=np.array([production, consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        mock_results = MagicMock()
        mock_parameters = {'number_of_families': 50, 'battery_size': 10}

        # Mock the PhysicalParameterCalculator to avoid None return
        with patch('parameteric_evaluation.time_aggregation_evaluation.PhysicalParameterCalculator') as mock_phys:
            mock_calc = MagicMock()
            mock_calc.calculate.return_value = (MagicMock(), 0.75)
            mock_phys.create.return_value = mock_calc

            # Just verify it doesn't crash
            result = TimeAggregationEvaluator.invoke(mock_dataset, mock_results, mock_parameters)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)


if __name__ == '__main__':
    unittest.main()
