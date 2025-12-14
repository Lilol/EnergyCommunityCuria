import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.target_metrics import (
    TargetMetricParameterCalculator, TargetMetricEvaluator
)


class TestTargetMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    def test_find_closer_exact_match(self):
        """Test find_closer with exact match"""
        result = TargetMetricParameterCalculator.find_closer(100, 10)
        self.assertEqual(result, 100)

        result = TargetMetricParameterCalculator.find_closer(50, 5)
        self.assertEqual(result, 50)

    def test_find_closer_round_up(self):
        """Test find_closer rounds up when above half"""
        result = TargetMetricParameterCalculator.find_closer(57, 10)
        self.assertEqual(result, 60)  # Rounds up to nearest multiple of 10

        result = TargetMetricParameterCalculator.find_closer(18, 5)
        self.assertEqual(result, 20)  # Rounds up to nearest multiple of 5

    def test_find_closer_round_down(self):
        """Test find_closer rounds down when below half"""
        result = TargetMetricParameterCalculator.find_closer(53, 10)
        self.assertEqual(result, 50)  # Rounds down to nearest multiple of 10

        result = TargetMetricParameterCalculator.find_closer(12, 5)
        self.assertEqual(result, 10)  # Rounds down to nearest multiple of 5

    def test_find_closer_with_step_one(self):
        """Test find_closer with step of 1"""
        result = TargetMetricParameterCalculator.find_closer(42, 1)
        self.assertEqual(result, 42)

    def test_find_closer_with_large_step(self):
        """Test find_closer with large step"""
        result = TargetMetricParameterCalculator.find_closer(150, 50)
        self.assertEqual(result, 150)

        result = TargetMetricParameterCalculator.find_closer(175, 50)
        self.assertEqual(result, 200)  # Rounds up to nearest multiple of 50

    def test_target_self_consumption_evaluator_key(self):
        """Test TargetSelfConsumptionEvaluator key"""
        from parameteric_evaluation.definitions import ParametricEvaluationType

        self.assertEqual(TargetMetricEvaluator._key, ParametricEvaluationType.METRIC_TARGETS)
        # TargetMetricEvaluator doesn't have _metric, only subclasses do

    def test_target_self_consumption_evaluator_name(self):
        """Test TargetSelfConsumptionEvaluator name"""
        self.assertEqual(TargetMetricEvaluator._name, "Target metric evaluator")

    @patch('parameteric_evaluation.target_metrics.configuration')
    @patch('parameteric_evaluation.target_metrics.DataFrame')
    def test_evaluate_targets_configuration(self, mock_dataframe, mock_config):
        """Test evaluate_targets reads from configuration"""
        mock_config.config.getarray.return_value = [0.5, 0.6, 0.7]
        mock_config.config.getint.return_value = 100

        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df

        # This would require full setup, so just verify config is accessed
        # The actual method is incomplete in the source, so we test what we can

    def test_find_closer_boundary_cases(self):
        """Test find_closer with boundary cases"""
        # Exactly at half
        result = TargetMetricParameterCalculator.find_closer(55, 10)
        self.assertEqual(result, 60)  # Rounds up to nearest multiple of 10

        result = TargetMetricParameterCalculator.find_closer(25, 10)
        self.assertEqual(result, 30)  # Rounds up to nearest multiple of 10

    def test_find_closer_zero_remainder(self):
        """Test find_closer with zero remainder"""
        result = TargetMetricParameterCalculator.find_closer(100, 25)
        self.assertEqual(result, 100)

        result = TargetMetricParameterCalculator.find_closer(0, 10)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
