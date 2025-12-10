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

    def test_time_aggregation_theoretical_limit(self):
        """Test time aggregation with theoretical limit (no aggregation)"""
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

    def test_time_aggregation_hourly(self):
        """Test hourly time aggregation"""
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

        self.assertIsInstance(value, (int, float, np.ndarray))

    def test_time_aggregation_daily(self):
        """Test daily time aggregation"""
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

    def test_time_aggregation_monthly(self):
        """Test monthly time aggregation"""
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

    def test_time_aggregation_yearly(self):
        """Test yearly time aggregation"""
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

    def test_time_aggregation_calculator_key(self):
        """Test TimeAggregationParameterCalculator key attribute"""
        class TestCalc(TimeAggregationParameterCalculator):
            _key = TimeAggregation.HOUR
            _aggregation = TimeAggregation.HOUR
            _metric = LoadMatchingMetric.SELF_CONSUMPTION

        self.assertEqual(TestCalc._key, TimeAggregation.HOUR)

    def test_time_aggregation_with_self_sufficiency(self):
        """Test time aggregation with self sufficiency metric"""
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

    @patch('parameteric_evaluation.time_aggregation_evaluation.plot_shared_energy')
    @patch('parameteric_evaluation.time_aggregation_evaluation.WriteDataArray')
    def test_time_aggregation_evaluator_invoke(self, mock_write, mock_plot):
        """Test TimeAggregationEvaluator invoke method"""
        from parameteric_evaluation.time_aggregation_evaluation import TimeAggregationEvaluator

        mock_dataset = MagicMock()
        mock_results = MagicMock()
        mock_parameters = {'number_of_families': 50, 'battery_size': 10}

        # Mock the parent invoke
        with patch.object(TimeAggregationEvaluator.__bases__[0], 'invoke', return_value=(mock_dataset, mock_results)):
            result = TimeAggregationEvaluator.invoke(mock_dataset, mock_results, mock_parameters)

        # Verify plot and write were called
        mock_plot.assert_called_once()
        mock_write.assert_called()


if __name__ == '__main__':
    unittest.main()

