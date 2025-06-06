from typing import Iterable

from xarray import DataArray

from data_storage.data_array_extensions import OmnesAccessor


class OmnesDataArray(DataArray):
    __slots__ = []

    @property
    def omnes(self):
        return OmnesAccessor(self)

    def update(self, data, coordinates):
        """
        Does not change the DataArray, please call dataarray = dataarray.update(), if you want to modify it.
        Update a cell in the OmnesDataArray using coordinate values.
        Adds missing dimensions and coordinates if they do not exist.

        Parameters:
        - data: scalar or Iterable[] value to assign
        - **coords: keyword arguments of dimension names and their coordinate values

        Returns:
        - Updated OmnesDataArray
        """

        # Normalize coordinates to lists
        normalized_coords = {dim: self.normalize(coord) for dim, coord in coordinates.items()}

        da = self
        # Expand or reindex to include all coords
        for dim, coords in normalized_coords.items():
            if dim not in da.dims:
                da = da.expand_dims({dim: coords})
            else:
                current_vals = set(da.coords[dim].values.tolist())
                new_vals = list(current_vals.union(coords))
                if len(new_vals) > len(current_vals):
                    da = da.reindex({dim: new_vals}, fill_value=0)

        # Ensure the DataArray is writable
        da = da.copy()

        # Assign the new data value, normalize the data to a list
        indexer = {dim: coord for dim, coord in coordinates.items()}
        da.loc[indexer] = OmnesDataArray.normalize_data(data)
        return da

    @staticmethod
    def normalize(coord):
        if isinstance(coord, DataArray):
            if coord.ndim == 0:
                return [coord.item()]
            else:
                return coord.values
        elif isinstance(coord, Iterable):
            return coord
        else:
            return [coord]

    @staticmethod
    def normalize_data(data):
        if isinstance(data, DataArray):
            if data.ndim == 0:
                return data.item()
            else:
                return data.values
        elif isinstance(data, Iterable):
            return [*data]
        return data
