import logging

import xarray as xr
from pandas import infer_freq, to_timedelta

from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import OtherParameters, PhysicalMetric

logger = logging.getLogger(__name__)


@xr.register_dataarray_accessor("omnes")
class OmnesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._original_time_resolution = None
        self._alias_map = {DataKind.PRODUCTION: OtherParameters.INJECTED_ENERGY,
                           PhysicalMetric.TOTAL_CONSUMPTION: OtherParameters.WITHDRAWN_ENERGY}

    def _infer_time_resolution(self, dim='time'):
        time = self._obj.coords.get(dim, None)
        if time is None:
            logger.info("Could not infer original time frequency.")
            return None
        try:
            freq = infer_freq(time.to_index())
        except ValueError | TypeError:
            freq = None
        return freq

    def sel(self, indexers=None, **kwargs):
        """
        Enhanced .sel() that resolves aliases before selecting.
        """
        indexers = indexers or {}
        combined = {**indexers, **kwargs}
        resolved = {dim: self._alias_map.get(sel_val, sel_val) for dim, sel_val in combined.items()}
        return self._obj.sel(resolved)

    def resample(self, freq: str, method: str = "mean", dim: str = "time"):
        """
        Resample the DataArray along the time dimension using the given frequency and method.

        Parameters:
        - freq: str — new frequency, e.g., '1H', '15min'
        - method: str — 'mean', 'sum', or interpolation method like 'linear', 'ffill', 'bfill'
        - dim: str — name of the time dimension (default: 'time')

        Returns:
        - Resampled DataArray
        """
        if dim not in self._obj.dims:
            raise ValueError(f"Dimension '{dim}' not found in the DataArray.")

        if self._original_time_resolution is None:
            self._original_time_resolution = self._infer_time_resolution(dim)

        # If we still can't infer, raise an error
        if self._original_time_resolution is None:
            raise ValueError(f"Could not infer time resolution for dimension '{dim}'")

        # Normalize frequency strings to have leading numbers (e.g., 'h' -> '1h', 'H' -> '1H')
        def normalize_freq(freq_str):
            if freq_str and not freq_str[0].isdigit():
                return '1' + freq_str
            return freq_str

        orig_freq_str = normalize_freq(self._original_time_resolution)
        target_freq_str = normalize_freq(freq)

        original_res = to_timedelta(orig_freq_str)
        target_res = to_timedelta(target_freq_str)

        rs = self._obj.resample({dim: freq})

        if target_res >= original_res:
            # Downsample or same resolution
            if method not in {"mean", "sum", "max", "min", "median"}:
                raise ValueError(f"Invalid downsampling method '{method}'.")
            result = getattr(rs, method)()
        else:
            # Upsample
            if method not in {"linear", "nearest", "ffill", "bfill"}:
                raise ValueError(f"Invalid upsampling method '{method}'.")
            if method in {"ffill", "bfill"}:
                result = rs.ffill() if method == "ffill" else rs.bfill()
            else:
                # Use interpolate with kind parameter instead of method to avoid conflict
                result = rs.interpolate(kind=method)

        return result
