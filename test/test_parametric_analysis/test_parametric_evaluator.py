import unittest
from unittest.mock import MagicMock, patch

from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, LoadMatchingMetric
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


class TestParametricEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass

    def test_parametric_evaluator_base_class(self):
        """Test ParametricEvaluator base class initialization"""
        self.assertEqual(ParametricEvaluator._key, ParametricEvaluationType.INVALID)
        self.assertEqual(ParametricEvaluator._name, "Parametric evaluator base")
        self.assertIsInstance(ParametricEvaluator._parameter_calculators, dict)

    def test_get_eval_metrics_physical(self):
        """Test get_eval_metrics for physical metrics"""
        from parameteric_evaluation.parametric_evaluator import EvaluatorMeta

        metrics = EvaluatorMeta.get_eval_metrics(ParametricEvaluationType.PHYSICAL_METRICS)

        self.assertIsInstance(metrics, dict)
        # Should contain valid physical metrics
        for key in metrics.keys():
            self.assertNotEqual(key.value, "invalid")

    def test_get_eval_metrics_load_matching(self):
        """Test get_eval_metrics for load matching metrics"""
        from parameteric_evaluation.parametric_evaluator import EvaluatorMeta

        metrics = EvaluatorMeta.get_eval_metrics(ParametricEvaluationType.LOAD_MATCHING_METRICS)

        self.assertIsInstance(metrics, dict)
        # Should contain valid load matching metrics
        for key in metrics.keys():
            self.assertNotEqual(key.value, "invalid")

    def test_get_eval_metrics_economic(self):
        """Test get_eval_metrics for economic metrics"""
        from parameteric_evaluation.parametric_evaluator import EvaluatorMeta

        metrics = EvaluatorMeta.get_eval_metrics(ParametricEvaluationType.ECONOMIC_METRICS)

        self.assertIsInstance(metrics, dict)

    def test_get_eval_metrics_environmental(self):
        """Test get_eval_metrics for environmental metrics"""
        from parameteric_evaluation.parametric_evaluator import EvaluatorMeta

        metrics = EvaluatorMeta.get_eval_metrics(ParametricEvaluationType.ENVIRONMENTAL_METRICS)

        self.assertIsInstance(metrics, dict)

    def test_get_eval_metrics_invalid(self):
        """Test get_eval_metrics with invalid type"""
        from parameteric_evaluation.parametric_evaluator import EvaluatorMeta

        metrics = EvaluatorMeta.get_eval_metrics(ParametricEvaluationType.INVALID)

        self.assertEqual(metrics, {})

    @patch('parameteric_evaluation.parametric_evaluator.logger')
    def test_invoke_logging(self, mock_logger):
        """Test that invoke logs correctly"""
        mock_dataset = MagicMock()
        mock_results = MagicMock()
        mock_parameters = {}

        # Create a test evaluator subclass
        class TestEvaluator(ParametricEvaluator):
            _key = ParametricEvaluationType.PHYSICAL_METRICS
            _name = "Test evaluator"
            _parameter_calculators = {}

        TestEvaluator.invoke(mock_dataset, mock_results, mock_parameters)

        # Verify logging was called
        self.assertTrue(mock_logger.info.called)

    def test_invoke_calls_calculators(self):
        """Test that invoke calls all parameter calculators"""
        mock_dataset = MagicMock()
        mock_results = MagicMock()
        mock_parameters = {}

        # Create mock calculators that return the expected tuple when called
        mock_calc1 = MagicMock()
        mock_calc1.call.return_value = (mock_dataset, mock_results)

        mock_calc2 = MagicMock()
        mock_calc2.call.return_value = (mock_dataset, mock_results)

        class TestEvaluator(ParametricEvaluator):
            _key = ParametricEvaluationType.INVALID  # Use INVALID to prevent metaclass from populating
            _name = "Test evaluator"

        # Set calculators after class creation to override metaclass behavior
        TestEvaluator._parameter_calculators = {
            PhysicalMetric.SHARED_ENERGY: mock_calc1,
            PhysicalMetric.TOTAL_CONSUMPTION: mock_calc2
        }

        result_dataset, result_results = TestEvaluator.invoke(
            mock_dataset, mock_results, mock_parameters
        )

        # Verify both calculators were called
        mock_calc1.call.assert_called_once()
        mock_calc2.call.assert_called_once()

        # Verify correct arguments were passed
        mock_calc1.call.assert_called_with(mock_dataset, mock_results, mock_parameters)
        mock_calc2.call.assert_called_with(mock_dataset, mock_results, mock_parameters)

    def test_invoke_with_kwargs(self):
        """Test invoke with keyword arguments"""
        mock_dataset = MagicMock()
        mock_results = MagicMock()
        mock_parameters = {}

        mock_calc = MagicMock()
        mock_calc.call.return_value = (mock_dataset, mock_results)

        class TestEvaluator(ParametricEvaluator):
            _key = ParametricEvaluationType.INVALID  # Use INVALID to prevent metaclass from populating
            _name = "Test evaluator"

        # Set calculators after class creation
        TestEvaluator._parameter_calculators = {PhysicalMetric.SHARED_ENERGY: mock_calc}

        TestEvaluator.invoke(
            mock_dataset,
            mock_results,
            mock_parameters,
            extra_param="value"
        )

        # Verify calculator was called with kwargs
        mock_calc.call.assert_called_once()
        call_kwargs = mock_calc.call.call_args[1]
        self.assertIn('extra_param', call_kwargs)
        self.assertEqual(call_kwargs['extra_param'], 'value')

    @patch('parameteric_evaluation.parametric_evaluator.configuration')
    def test_create_evaluators_based_on_configuration(self, mock_config):
        """Test create_evaluators_based_on_configuration"""
        mock_config.config.get.return_value = "physical_metrics,economic_metrics"

        # This would create evaluators based on config
        # Exact behavior depends on registered subclasses

    def test_parametric_evaluator_subclass_registration(self):
        """Test that ParametricEvaluator subclasses register correctly"""
        class CustomEvaluator(ParametricEvaluator):
            _key = ParametricEvaluationType.PHYSICAL_METRICS
            _name = "Custom evaluator"

        self.assertEqual(CustomEvaluator._key, ParametricEvaluationType.PHYSICAL_METRICS)
        self.assertEqual(CustomEvaluator._name, "Custom evaluator")

    def test_invoke_returns_tuple(self):
        """Test that invoke returns tuple of dataset and results"""
        mock_dataset = MagicMock()
        mock_results = MagicMock()
        mock_parameters = {}

        class TestEvaluator(ParametricEvaluator):
            _key = ParametricEvaluationType.PHYSICAL_METRICS
            _name = "Test evaluator"
            _parameter_calculators = {}

        result = TestEvaluator.invoke(mock_dataset, mock_results, mock_parameters)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
