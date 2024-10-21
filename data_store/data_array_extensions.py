import xarray as xr


@xr.register_dataset_accessor("omnes")
class OmnesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._original_time_resolution = None

    @property
    def sel(self):
        pass

    @property
    def center(self):
        """Return the geographic center point of this dataset."""
        if self._original_time_resolution is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
        return self._original_time_resolution
