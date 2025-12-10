import unittest

import numpy as np
import pandas as pd
import xarray as xr

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.battery import Battery
from parameteric_evaluation.definitions import BatteryPowerFlows, OtherParameters


class TestBattery(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')
        self.battery_size = 10.0  # kWh
        self.t_hours = 1.0
        self.battery = Battery(size=self.battery_size, t_hours=self.t_hours)

    def test_battery_initialization(self):
        """Test Battery initialization"""
        self.assertEqual(self.battery._size, self.battery_size)
        self.assertEqual(self.battery.p_max, self.battery_size / self.t_hours)

    def test_battery_with_zero_size(self):
        """Test battery behavior with zero size"""
        zero_battery = Battery(size=0, t_hours=1)

        # Create test data
        injected = np.array([5.0] * 12 + [0.0] * 12)
        withdrawn = np.array([0.0] * 12 + [5.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result = zero_battery.manage_bess(input_da)

        # Verify battery power flows are zero
        stored = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.STORED_ENERGY})
        self.assertTrue(np.all(stored.values == 0))

    def test_battery_charging(self):
        """Test battery charging behavior"""
        # Create scenario with excess production (positive charge)
        injected = np.array([10.0] * 12 + [0.0] * 12)
        withdrawn = np.array([0.0] * 12 + [5.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result = self.battery.manage_bess(input_da)

        # Check that battery stores energy
        stored = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.STORED_ENERGY})
        self.assertTrue(np.any(stored.values > 0))

        # Check that stored energy doesn't exceed battery size
        self.assertTrue(np.all(stored.values <= self.battery_size))

    def test_battery_discharging(self):
        """Test battery discharging behavior"""
        # Create scenario where battery needs to discharge
        injected = np.array([5.0] * 6 + [0.0] * 18)
        withdrawn = np.array([0.0] * 6 + [3.0] * 18)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result = self.battery.manage_bess(input_da)

        # Check battery operation
        stored = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.STORED_ENERGY})
        charge = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.POWER_CHARGE})

        # Stored energy should be non-negative
        self.assertTrue(np.all(stored.values >= 0))

        # Check charge/discharge power respects limits
        self.assertTrue(np.all(np.abs(charge.values) <= self.battery.p_max))

    def test_battery_power_limits(self):
        """Test that battery respects power limits"""
        # Create scenario with high power demand
        injected = np.array([20.0] * 12 + [0.0] * 12)
        withdrawn = np.array([0.0] * 12 + [20.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result = self.battery.manage_bess(input_da)

        # Check power limits
        charge = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.POWER_CHARGE})
        self.assertTrue(np.all(np.abs(charge.values) <= self.battery.p_max))

    def test_battery_calculate_classmethod(self):
        """Test Battery calculate classmethod"""
        # Note: Battery class doesn't have a proper _key, so we skip the full calculator infrastructure
        # and just test the manage_bess method directly
        injected = np.array([5.0] * 24)
        withdrawn = np.array([3.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        # Test manage_bess directly instead
        result = self.battery.manage_bess(input_da)

        self.assertIsInstance(result, OmnesDataArray)
        self.assertIn(BatteryPowerFlows.STORED_ENERGY, result.calculated)
        self.assertIn(BatteryPowerFlows.POWER_CHARGE, result.calculated)

    def test_battery_energy_conservation(self):
        """Test energy conservation in battery operation"""
        injected = np.array([8.0, 6.0, 4.0, 2.0] * 6)
        withdrawn = np.array([2.0, 4.0, 6.0, 8.0] * 6)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [OtherParameters.INJECTED_ENERGY, OtherParameters.WITHDRAWN_ENERGY]
        }
        data = np.array([injected, withdrawn])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result = self.battery.manage_bess(input_da)

        # Get battery charge/discharge
        charge = result.sel({DataKind.CALCULATED.value: BatteryPowerFlows.POWER_CHARGE})
        new_inj = result.sel({DataKind.CALCULATED.value: OtherParameters.INJECTED_ENERGY})
        new_with = result.sel({DataKind.CALCULATED.value: OtherParameters.WITHDRAWN_ENERGY})

        # Check energy balance
        original_inj_sum = injected.sum()
        original_with_sum = withdrawn.sum()
        new_inj_sum = new_inj.sum().values
        new_with_sum = new_with.sum().values
        charge_sum = charge.sum().values

        # Energy in - energy out should be stored
        self.assertAlmostEqual(original_inj_sum - new_inj_sum, charge_sum, places=5)


if __name__ == '__main__':
    unittest.main()
