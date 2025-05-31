import xarray as xr

from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import OtherParameters, PhysicalMetric


@xr.register_dataset_accessor("omnes")
class OmnesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._original_time_resolution = None
        self._alias_map = {DataKind.PRODUCTION: OtherParameters.INJECTED_ENERGY,
                           PhysicalMetric.TOTAL_CONSUMPTION: OtherParameters.WITHDRAWN_ENERGY}

    @property
    def sel(self, indexers=None, **kwargs):
        """
        Enhanced .sel() that resolves aliases before selecting.
        """
        indexers = indexers or {}
        combined = {**indexers, **kwargs}
        resolved = {
            dim: self._alias_map.get(sel_val, sel_val)
            for dim, sel_val in combined.items()
        }
        return self._obj.sel(resolved)

    @property
    def center(self):
        """Return the geographic center point of this dataset."""
        if self._original_time_resolution is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
        return self._original_time_resolution
