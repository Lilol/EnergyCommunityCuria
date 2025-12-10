import unittest

from parameteric_evaluation.dimensions import convert_to_hours, power_to_energy


class TestDimensions(unittest.TestCase):
    def test_convert_to_hours_from_hours(self):
        """Test conversion from hours to hours"""
        self.assertEqual(convert_to_hours('1h'), 1.0)
        self.assertEqual(convert_to_hours('2h'), 2.0)
        self.assertEqual(convert_to_hours('0.5h'), 0.5)

    def test_convert_to_hours_from_minutes(self):
        """Test conversion from minutes to hours"""
        self.assertEqual(convert_to_hours('60min'), 1.0)
        self.assertEqual(convert_to_hours('30min'), 0.5)
        self.assertEqual(convert_to_hours('15min'), 0.25)
        self.assertEqual(convert_to_hours('90min'), 1.5)

    def test_convert_to_hours_from_seconds(self):
        """Test conversion from seconds to hours"""
        self.assertEqual(convert_to_hours('3600s'), 1.0)
        self.assertEqual(convert_to_hours('1800s'), 0.5)
        self.assertEqual(convert_to_hours('3600sec'), 1.0)

    def test_convert_to_hours_with_spaces(self):
        """Test conversion with leading/trailing spaces"""
        self.assertEqual(convert_to_hours('  1h  '), 1.0)
        self.assertEqual(convert_to_hours(' 30min '), 0.5)

    def test_convert_to_hours_case_insensitive(self):
        """Test conversion is case insensitive"""
        self.assertEqual(convert_to_hours('1H'), 1.0)
        self.assertEqual(convert_to_hours('30MIN'), 0.5)
        self.assertEqual(convert_to_hours('3600SEC'), 1.0)

    def test_convert_to_hours_invalid_format(self):
        """Test conversion with invalid format raises ValueError"""
        with self.assertRaises(ValueError):
            convert_to_hours('invalid')

        with self.assertRaises(ValueError):
            convert_to_hours('1day')

        with self.assertRaises(ValueError):
            convert_to_hours('1')

    def test_power_to_energy_default_dt(self):
        """Test power_to_energy with default dt"""
        # Assuming default dt is set from config
        power = 100.0
        energy = power_to_energy(power)

        # Energy should be power * dt
        self.assertIsInstance(energy, (int, float))
        self.assertGreaterEqual(energy, 0)

    def test_power_to_energy_with_custom_dt(self):
        """Test power_to_energy with custom dt"""
        power = 100.0
        dt = 0.5  # 30 minutes
        energy = power_to_energy(power, dt)

        self.assertEqual(energy, 50.0)

    def test_power_to_energy_with_one_hour(self):
        """Test power_to_energy with 1 hour dt"""
        power = 100.0
        dt = 1.0
        energy = power_to_energy(power, dt)

        self.assertEqual(energy, 100.0)

    def test_power_to_energy_with_zero_power(self):
        """Test power_to_energy with zero power"""
        power = 0.0
        dt = 1.0
        energy = power_to_energy(power, dt)

        self.assertEqual(energy, 0.0)

    def test_power_to_energy_with_array(self):
        """Test power_to_energy with array input"""
        import numpy as np

        power = np.array([100.0, 200.0, 300.0])
        dt = 0.25  # 15 minutes
        energy = power_to_energy(power, dt)

        expected = np.array([25.0, 50.0, 75.0])
        np.testing.assert_array_almost_equal(energy, expected)

    def test_power_to_energy_with_quarter_hour(self):
        """Test power_to_energy with 15-minute intervals"""
        power = 1000.0
        dt = 0.25
        energy = power_to_energy(power, dt)

        self.assertEqual(energy, 250.0)

    def test_convert_to_hours_fractional_values(self):
        """Test conversion with fractional values"""
        self.assertEqual(convert_to_hours('0.25h'), 0.25)
        self.assertEqual(convert_to_hours('1.5h'), 1.5)
        self.assertEqual(convert_to_hours('45min'), 0.75)

    def test_power_to_energy_negative_power(self):
        """Test power_to_energy with negative power (discharge)"""
        power = -100.0
        dt = 1.0
        energy = power_to_energy(power, dt)

        self.assertEqual(energy, -100.0)


if __name__ == '__main__':
    unittest.main()

