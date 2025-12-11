import unittest
from unittest.mock import MagicMock, patch

from io_operation.input.definitions import DataKind
from parameteric_evaluation.parameter_pack import EvaluationParameterPack


class TestParameterPack(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass

    def test_simple_parameter_pack(self):
        """Test parameter pack with simple list inputs"""
        params_str = "{'battery_size': [0, 10, 20], 'number_of_families': [10, 20, 30]}"

        pack = EvaluationParameterPack(parameters=params_str)

        self.assertEqual(pack.bess_sizes, [0, 10, 20])
        self.assertEqual(pack.number_of_families, [10, 20, 30])

    def test_parameter_pack_iteration(self):
        """Test iterating over parameter pack"""
        params_str = "{'battery_size': [0, 10], 'number_of_families': [10, 20]}"

        pack = EvaluationParameterPack(parameters=params_str)

        combinations = list(pack)

        # Should have 2 * 2 = 4 combinations
        self.assertEqual(len(combinations), 4)

        # Each combination should have both keys
        for combo in combinations:
            self.assertIn('battery_size', combo)
            self.assertIn('number_of_families', combo)

    def test_parameter_pack_nested_dict(self):
        """Test parameter pack with nested dictionary"""
        params_str = "{'battery_size': {0: [10, 20], 10: [30, 40]}}"

        pack = EvaluationParameterPack(parameters=params_str)

        # Should extract all battery sizes and family numbers
        self.assertIn(0, pack.bess_sizes)
        self.assertIn(10, pack.bess_sizes)

    def test_convert_to_int_vector(self):
        """Test convert_to_int_vector static method"""
        values = ['1', '2', '3', '4']
        result = EvaluationParameterPack.convert_to_int_vector(values)

        self.assertEqual(result, [1, 2, 3, 4])
        self.assertIsInstance(result[0], int)

    def test_parameter_pack_with_zero_battery(self):
        """Test parameter pack with zero battery size"""
        params_str = "{'battery_size': [0], 'number_of_families': [50]}"

        pack = EvaluationParameterPack(parameters=params_str)

        combinations = list(pack)

        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0]['battery_size'], 0)
        self.assertEqual(combinations[0]['number_of_families'], 50)

    def test_parameter_pack_single_values(self):
        """Test parameter pack with single values"""
        params_str = "{'battery_size': [10], 'number_of_families': [20]}"

        pack = EvaluationParameterPack(parameters=params_str)

        combinations = list(pack)

        self.assertEqual(len(combinations), 1)

    def test_parameter_pack_large_combinations(self):
        """Test parameter pack with many combinations"""
        params_str = "{'battery_size': [0, 5, 10, 15, 20], 'number_of_families': [10, 20, 30, 40, 50]}"

        pack = EvaluationParameterPack(parameters=params_str)

        combinations = list(pack)

        # Should have 5 * 5 = 25 combinations
        self.assertEqual(len(combinations), 25)

    @patch('parameteric_evaluation.parameter_pack.configuration')
    def test_parameter_pack_from_config(self, mock_config):
        """Test parameter pack initialization from config"""
        mock_config.config.getstr.return_value = "{'battery_size': [10], 'number_of_families': [20]}"

        pack = EvaluationParameterPack()

        mock_config.config.getstr.assert_called_once()

    def test_parameter_pack_incomplete_pairing_error(self):
        """Test parameter pack with incomplete pairing raises error"""
        params_str = "{'battery_size': {0: [10, 20]}}"

        # Should raise RuntimeError for incomplete specification
        with self.assertRaises(RuntimeError):
            pack = EvaluationParameterPack(parameters=params_str)

    def test_collect_combinations_basic(self):
        """Test collect_combinations_from_non_complete_pairing"""
        params = {
            DataKind.BATTERY_SIZE.value: [0, 10, 20],
            DataKind.NUMBER_OF_FAMILIES.value: [10, 20, 30]
        }

        bess, families, combos = EvaluationParameterPack.collect_combinations_from_non_complete_pairing(params)

        self.assertIsInstance(bess, set)
        self.assertIsInstance(families, set)
        self.assertIsInstance(combos, list)

    def test_parameter_pack_yields_correct_format(self):
        """Test that parameter pack yields correctly formatted dicts"""
        params_str = "{'battery_size': [10], 'number_of_families': [20]}"

        pack = EvaluationParameterPack(parameters=params_str)

        for combo in pack:
            self.assertIsInstance(combo, dict)
            self.assertIn('battery_size', combo)
            self.assertIn('number_of_families', combo)
            self.assertIsInstance(combo['battery_size'], (int, float))
            self.assertIsInstance(combo['number_of_families'], (int, float))


if __name__ == '__main__':
    unittest.main()

