import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import PhysicalMetric


class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    def test_calculator_base_class_initialization(self):
        """Test Calculator base class initialization"""
        self.assertEqual(Calculator._key, PhysicalMetric)
        self.assertEqual(Calculator._name, "calculator")

    def test_calculator_calculate_not_implemented(self):
        """Test that base Calculator.calculate raises NotImplementedError"""
        with self.assertRaises(NotImplementedError):
            Calculator.calculate()

    def test_calculator_postprocess_with_none(self):
        """Test postprocess with None inputs"""
        result = Calculator.postprocess(None, None, {})
        self.assertIsNone(result)

        result = Calculator.postprocess("data", None, {})
        self.assertIsNone(result)

    def test_calculator_postprocess_with_data(self):
        """Test postprocess with valid data"""
        # Create mock data
        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY]
        }
        mock_result = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        mock_previous = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # Mock update method
        mock_previous.update = MagicMock(return_value=mock_previous)

        parameters = {"test_param": "value"}
        result = Calculator.postprocess(mock_result, mock_previous, parameters)

        # Verify update was called
        mock_previous.update.assert_called_once()

    def test_calculator_postprocess_with_tuple(self):
        """Test postprocess with tuple result"""
        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY]
        }
        mock_data = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        mock_previous = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # Mock update method
        mock_previous.update = MagicMock(return_value=mock_previous)

        result_tuple = (None, mock_data)
        parameters = {"test_param": "value"}

        result = Calculator.postprocess(result_tuple, mock_previous, parameters)

        # Verify update was called
        mock_previous.update.assert_called_once()

    def test_calculator_call_method(self):
        """Test Calculator.call method"""
        # Create a mock calculator subclass
        class MockCalculator(Calculator):
            _key = PhysicalMetric.SHARED_ENERGY

            @classmethod
            def calculate(cls, input_da=None, results_of_previous_calculations=None, **kwargs):
                return input_da, results_of_previous_calculations

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.SHARED_ENERGY]
        }
        input_da = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        results = OmnesDataArray(
            data=np.random.rand(1, 24),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        parameters = {"param": "value"}

        result_input, result_output = MockCalculator.call(input_da, results, parameters)

        self.assertIsNotNone(result_input)
        self.assertIsNotNone(result_output)

    def test_calculator_subclass_registration(self):
        """Test that Calculator subclasses are registered"""
        class TestCalc(Calculator):
            _key = PhysicalMetric.TOTAL_CONSUMPTION

            @classmethod
            def calculate(cls, input_da=None, results_of_previous_calculations=None, **kwargs):
                return input_da, results_of_previous_calculations

        # Verify subclass exists
        self.assertEqual(TestCalc._key, PhysicalMetric.TOTAL_CONSUMPTION)

    @patch('parameteric_evaluation.calculator.logger')
    def test_calculator_logging(self, mock_logger):
        """Test that calculator logging works"""
        class LoggingCalc(Calculator):
            _key = PhysicalMetric.SHARED_ENERGY

            @classmethod
            def calculate(cls, input_da=None, results_of_previous_calculations=None, **kwargs):
                return input_da, results_of_previous_calculations

        # Call calculate to trigger logging
        LoggingCalc.calculate()

        # Verify logger was called
        mock_logger.info.assert_called()


if __name__ == '__main__':
    unittest.main()

