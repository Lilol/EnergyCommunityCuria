from xarray import DataArray


class OmnesDataset(DataArray):
    def __init__(self, data):
        super().__init__(data=data)
