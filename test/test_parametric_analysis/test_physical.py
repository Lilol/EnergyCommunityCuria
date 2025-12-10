import unittest

import numpy as np
import pandas as pd
import xarray as xr

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import PhysicalMetric, OtherParameters
from parameteric_evaluation.physical import SharedEnergy, TotalConsumption


class TestPhysical(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    def test_shared_energy_with_production_consumption(self):
        """Test SharedEnergy calculation with production and consumption"""
        production = np.array([10.0] * 12 + [5.0] * 12)
        consumption = np.array([5.0] * 12 + [10.0] * 12)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION]
        }
        data = np.array([production, consumption])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = SharedEnergy.calculate(input_da)

        # Verify shared energy is minimum of production and consumption
        shared = result.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        self.assertTrue(np.all(shared.values[:12] == 5.0))
        self.assertTrue(np.all(shared.values[12:] == 5.0))

    def test_shared_energy_with_injected_withdrawn(self):
        """Test SharedEnergy calculation with injected and withdrawn energy"""
        injected = np.array([8.0] * 12 + [3.0] * 12)
        withdrawn = np.array([4.0] * 12 + [9.0] * 12)

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

        result, _ = SharedEnergy.calculate(input_da)

        # Verify shared energy is minimum of injected and withdrawn
        shared = result.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        self.assertTrue(np.all(shared.values[:12] == 4.0))
        self.assertTrue(np.all(shared.values[12:] == 3.0))

    def test_shared_energy_missing_indices(self):
        """Test SharedEnergy raises error with missing indices"""
        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION]  # Missing consumption
        }
        input_da = OmnesDataArray(
            data=np.array([np.random.rand(24)]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        with self.assertRaises(IndexError):
            SharedEnergy.calculate(input_da)

    def test_shared_energy_recalculation(self):
        """Test SharedEnergy recalculation when already present"""
        production = np.array([10.0] * 24)
        consumption = np.array([5.0] * 24)
        old_shared = np.array([3.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, DataKind.CONSUMPTION, PhysicalMetric.SHARED_ENERGY]
        }
        data = np.array([production, consumption, old_shared])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = SharedEnergy.calculate(input_da)

        # Verify shared energy was recalculated
        shared = result.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})
        self.assertTrue(np.all(shared.values == 5.0))

    def test_total_consumption_calculation(self):
        """Test TotalConsumption calculation"""
        families_consumption = np.array([2.0] * 24)
        users_consumption = np.array([3.0] * 24)
        number_of_families = 10

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.CONSUMPTION_OF_FAMILIES, DataKind.CONSUMPTION_OF_USERS]
        }
        data = np.array([families_consumption, users_consumption])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = TotalConsumption.calculate(input_da, number_of_families=number_of_families)

        # Verify total consumption = families * n + users
        total = result.sel({DataKind.CALCULATED.value: PhysicalMetric.TOTAL_CONSUMPTION})
        expected = families_consumption * number_of_families + users_consumption
        np.testing.assert_array_almost_equal(total.values, expected)

    def test_total_consumption_with_zero_families(self):
        """Test TotalConsumption with zero families"""
        families_consumption = np.array([2.0] * 24)
        users_consumption = np.array([3.0] * 24)
        number_of_families = 0

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.CONSUMPTION_OF_FAMILIES, DataKind.CONSUMPTION_OF_USERS]
        }
        data = np.array([families_consumption, users_consumption])
        input_da = OmnesDataArray(
            data=data,
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = TotalConsumption.calculate(input_da, number_of_families=number_of_families)

        # Verify total consumption equals only users consumption
        total = result.sel({DataKind.CALCULATED.value: PhysicalMetric.TOTAL_CONSUMPTION})
        np.testing.assert_array_almost_equal(total.values, users_consumption)

    def test_shared_energy_key(self):
        """Test SharedEnergy calculator key"""
        self.assertEqual(SharedEnergy._key, PhysicalMetric.SHARED_ENERGY)

    def test_total_consumption_key(self):
        """Test TotalConsumption calculator key"""
        self.assertEqual(TotalConsumption._key, PhysicalMetric.TOTAL_CONSUMPTION)

    def test_shared_energy_postprocess(self):
        """Test SharedEnergy postprocess returns correct value"""
        result = SharedEnergy.postprocess(None, None, {})
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

