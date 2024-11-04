from xarray import DataArray


class OmnesDataArray(DataArray):
    def __init__(self, data):
        super().__init__(data=data)
