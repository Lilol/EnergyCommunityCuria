import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import OtherParameters, PhysicalMetric
from parameteric_evaluation.other_calculators import InjectedEnergy, WithdrawnEnergy, Equality


class TestOtherCalculators(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.time = pd.date_range(start='2023-01-01', periods=24, freq='h')

    def test_injected_energy_equality(self):
        """Test InjectedEnergy creates alias for PRODUCTION"""
        production = np.array([10.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = InjectedEnergy.calculate(input_da)

        # Verify that INJECTED_ENERGY now appears in coordinates
        self.assertIn(OtherParameters.INJECTED_ENERGY, result.coords[DataKind.CALCULATED.value].values)

    def test_withdrawn_energy_equality(self):
        """Test WithdrawnEnergy creates alias for TOTAL_CONSUMPTION"""
        consumption = np.array([15.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.TOTAL_CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = WithdrawnEnergy.calculate(input_da)

        # Verify that WITHDRAWN_ENERGY now appears in coordinates
        self.assertIn(OtherParameters.WITHDRAWN_ENERGY, result.coords[DataKind.CALCULATED.value].values)

    def test_injected_energy_already_exists(self):
        """Test InjectedEnergy when already exists"""
        production = np.array([10.0] * 24)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION, OtherParameters.INJECTED_ENERGY]
        }
        input_da = OmnesDataArray(
            data=np.array([production, production]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = InjectedEnergy.calculate(input_da)

        # Should return input unchanged - use xarray's equals method
        self.assertTrue(result.equals(input_da))

    def test_withdrawn_energy_missing_source(self):
        """Test WithdrawnEnergy when source doesn't exist"""
        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION]
        }
        input_da = OmnesDataArray(
            data=np.array([np.random.rand(24)]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = WithdrawnEnergy.calculate(input_da)

        # Should return input unchanged since source doesn't exist - use xarray's equals method
        self.assertTrue(result.equals(input_da))

    def test_equality_keys(self):
        """Test Equality subclass keys"""
        self.assertEqual(InjectedEnergy._key, OtherParameters.INJECTED_ENERGY)
        self.assertEqual(WithdrawnEnergy._key, OtherParameters.WITHDRAWN_ENERGY)

        self.assertEqual(InjectedEnergy._equate_to, DataKind.PRODUCTION)
        self.assertEqual(WithdrawnEnergy._equate_to, PhysicalMetric.TOTAL_CONSUMPTION)

    def test_injected_energy_coordinate_replacement(self):
        """Test InjectedEnergy replaces coordinate correctly"""
        production = np.array([5.0, 10.0, 15.0, 20.0] * 6)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [DataKind.PRODUCTION]
        }
        input_da = OmnesDataArray(
            data=np.array([production]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = InjectedEnergy.calculate(input_da)

        # Check that PRODUCTION is replaced with INJECTED_ENERGY
        calc_coords = result.coords[DataKind.CALCULATED.value].values
        self.assertIn(OtherParameters.INJECTED_ENERGY, calc_coords)

        # Verify data is preserved
        injected = result.sel({DataKind.CALCULATED.value: OtherParameters.INJECTED_ENERGY})
        np.testing.assert_array_almost_equal(injected.values, production)

    def test_withdrawn_energy_coordinate_replacement(self):
        """Test WithdrawnEnergy replaces coordinate correctly"""
        consumption = np.array([8.0, 12.0, 16.0, 20.0] * 6)

        coords = {
            DataKind.TIME.value: self.time,
            DataKind.CALCULATED.value: [PhysicalMetric.TOTAL_CONSUMPTION]
        }
        input_da = OmnesDataArray(
            data=np.array([consumption]),
            dims=[DataKind.CALCULATED.value, DataKind.TIME.value],
            coords=coords
        )

        result, _ = WithdrawnEnergy.calculate(input_da)

        # Check that TOTAL_CONSUMPTION is replaced with WITHDRAWN_ENERGY
        calc_coords = result.coords[DataKind.CALCULATED.value].values
        self.assertIn(OtherParameters.WITHDRAWN_ENERGY, calc_coords)

        # Verify data is preserved
        withdrawn = result.sel({DataKind.CALCULATED.value: OtherParameters.WITHDRAWN_ENERGY})
        np.testing.assert_array_almost_equal(withdrawn.values, consumption)

    def test_equality_postprocess(self):
        """Test Equality postprocess returns None for results"""
        result = Equality.calculate(None)
        self.assertEqual(result, (None, None))


if __name__ == '__main__':
    unittest.main()
