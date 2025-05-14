from xarray import DataArray


class OmnesDataArray(DataArray):
    __slots__ = []

    def update(self, data, **coordinates):
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
        da = self
        for dim, coord in coordinates.items():
            if dim not in da.dims:
                da = da.expand_dims({dim: [coord]})
            elif coord not in da.coords[dim].values:
                new_coords = da.coords[dim].values.tolist() + [coord]
                da = da.reindex({dim: new_coords}, fill_value=0)

        # Ensure the DataArray is writable
        da = da.copy()

        # Assign the new data value
        indexer = {dim: coord for dim, coord in coordinates.items()}
        da.loc[indexer] = data
        return da
