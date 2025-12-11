import unittest

import numpy as np
import pandas as pd
import xarray as xr
from hypothesis import given, strategies as st

from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import OtherParameters, PhysicalMetric


class TestOmnesAccessor(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with known values
        self.time = pd.date_range(start='2023-01-01', periods=12, freq='h')
        self.da = OmnesDataArray(data=np.arange(12), dims=['time'], coords={'time': ('time', self.time)}, name='test_da',
            attrs={'custom_attr': 'value'})
        self.accessor = self.da.omnes

    def test_init(self):
        """Test accessor initialization"""
        self.assertEqual(self.accessor._obj.name, 'test_da')
        self.assertEqual(self.accessor._original_time_resolution, None)
        self.assertEqual(self.accessor._alias_map, {DataKind.PRODUCTION: OtherParameters.INJECTED_ENERGY,
            PhysicalMetric.TOTAL_CONSUMPTION: OtherParameters.WITHDRAWN_ENERGY})

    def test_infer_time_resolution(self):
        """Test time resolution inference"""
        freq = self.accessor._infer_time_resolution()
        self.assertEqual(freq, 'h')  # Should detect hourly frequency

        # Test case where time coordinate is missing
        da_no_time = xr.DataArray(np.arange(5))
        accessor_no_time = da_no_time.omnes
        freq = accessor_no_time._infer_time_resolution()
        self.assertIsNone(freq)

    def test_sel_basic(self):
        """Test basic selection functionality"""
        # Test direct selection
        result = self.accessor.sel(time=self.time[0])
        self.assertEqual(result.values.item(), 0)

        # Test with 'calculated' dimension - select by the actual coordinate value
        da_with_calc = OmnesDataArray(
            data=np.arange(12).reshape(1, 12),
            dims=['calculated', 'time'],
            coords={
                'calculated': [DataKind.PRODUCTION],
                'time': self.time
            }
        )
        accessor_calc = da_with_calc.omnes
        # Select using the calculated dimension with the enum value
        result = accessor_calc.sel(calculated=DataKind.PRODUCTION, method='nearest')
        self.assertEqual(result.shape, (12,))

    def test_sel_complex(self):
        """Test complex selection scenarios"""
        # Test multiple dimensions
        da_multi = xr.DataArray(np.zeros((2, 3)), dims=['x', 'y'], coords={'x': [1, 2], 'y': ['a', 'b', 'c']})
        accessor_multi = da_multi.omnes

        result = accessor_multi.sel(x=1, y='a')
        self.assertEqual(result.values.item(), 0)

        # Test with actual aliasable dimensions
        da_with_aliases = OmnesDataArray(
            data=np.ones((2, 2)),
            dims=['calculated', 'time'],
            coords={
                'calculated': [DataKind.PRODUCTION, PhysicalMetric.TOTAL_CONSUMPTION],
                'time': [self.time[0], self.time[1]]
            }
        )
        accessor_aliases = da_with_aliases.omnes
        # Select using the actual dimension names with method='nearest' for enum coords
        result = accessor_aliases.sel(calculated=DataKind.PRODUCTION, time=self.time[0], method='nearest')
        self.assertEqual(result.values.item(), 1.0)

    def test_sel_errors(self):
        """Test selection error handling"""
        # Test invalid dimension - should raise KeyError from xarray, not ValueError
        with self.assertRaises((ValueError, KeyError)):
            self.accessor.sel(invalid_dim='value')

        # Test invalid value
        with self.assertRaises(KeyError):
            self.accessor.sel(time='invalid_time')

    @given(freq=st.sampled_from(['H', '2H', '4H']), method=st.sampled_from(['mean', 'sum', 'max']))
    def test_resample_downsample(self, freq, method):
        """Test downsampling functionality"""
        result = self.accessor.resample(freq=freq, method=method)

        # Verify dimension size changed correctly
        # Extract the numeric part from freq (e.g., '2H' -> 2, 'H' -> 1)
        import re
        freq_num = int(re.findall(r'\d+', freq)[0]) if re.findall(r'\d+', freq) else 1
        expected_size = len(self.time) // freq_num
        self.assertEqual(len(result.time), expected_size)

        # Verify values are aggregated correctly
        if method == 'mean':
            expected = np.mean(np.arange(12).reshape(-1, freq_num), axis=1)
        elif method == 'sum':
            expected = np.sum(np.arange(12).reshape(-1, freq_num), axis=1)
        else:  # max
            expected = np.max(np.arange(12).reshape(-1, freq_num), axis=1)

        np.testing.assert_array_almost_equal(result.values, expected)

    @given(method=st.sampled_from(['linear', 'nearest', 'ffill']))
    def test_resample_upsample(self, method):
        """Test upsampling functionality"""
        result = self.accessor.resample(freq='15min', method=method)

        # Verify dimension size increased - note: pandas may not include the endpoint
        # For 12 hourly samples from 00:00 to 11:00, upsampling to 15min gives:
        # 00:00, 00:15, 00:30, 00:45 for each hour = 4 per hour * 11 hours + 1 = 45 points
        expected_size = (len(self.time) - 1) * 4 + 1
        self.assertEqual(len(result.time), expected_size)

        # Verify interpolation methods
        if method == 'linear':
            # Linear interpolation should preserve monotonicity
            self.assertTrue(np.all(np.diff(result.values) >= 0))
        elif method == 'nearest':
            # Nearest neighbor should only contain original values
            np.testing.assert_array_equal(np.unique(result.values), np.arange(12))

    def test_resample_errors(self):
        """Test resampling error handling"""
        # Test invalid frequency
        with self.assertRaises(ValueError):
            self.accessor.resample(freq='invalid')

        # Test invalid method for downsampling
        with self.assertRaises(ValueError):
            self.accessor.resample(freq='2H', method='invalid')

        # Test invalid method for upsampling
        with self.assertRaises(ValueError):
            self.accessor.resample(freq='15min', method='invalid')

        # Test missing dimension
        with self.assertRaises(ValueError):
            self.accessor.resample(freq='H', dim='invalid_dim')


if __name__ == '__main__':
    unittest.main()
