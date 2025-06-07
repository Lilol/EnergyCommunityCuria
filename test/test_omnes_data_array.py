import numpy as np
import xarray as xr
import pytest

from data_storage.data_array_extensions import OmnesAccessor
from data_storage.omnes_data_array import OmnesDataArray


@pytest.fixture
def base_oda():
    data = xr.DataArray([1.0], coords={"time": ["2024-01-01"]}, dims="time")
    return OmnesDataArray(data)


def test_update_add_new_dimension(base_oda):
    updated = base_oda.update(3.14, {"metric": "voltage"})
    assert "metric" in updated.dims
    assert updated.sel(metric="voltage").item() == 3.14


def test_update_add_new_coordinate(base_oda):
    base = base_oda.expand_dims({"metric": ["voltage"]})
    oda = OmnesDataArray(base)
    updated = oda.update(2.5, {"time": "2024-01-02", "metric": "voltage"})
    assert "2024-01-02" in updated.coords["time"].values
    assert updated.sel(time="2024-01-02", metric="voltage").item() == 2.5


def test_update_overwrite_existing_value(base_oda):
    updated = base_oda.update(42.0, {"time": "2024-01-01"})
    assert updated.sel(time="2024-01-01").item() == 42.0


def test_normalize_scalar():
    assert OmnesDataArray.normalize(5) == [5]


def test_normalize_array():
    da = xr.DataArray([1, 2, 3])
    assert np.array_equal(OmnesDataArray.normalize(da), np.array([1, 2, 3]))


def test_normalize_data_scalar():
    assert OmnesDataArray.normalize_data(5) == 5


def test_normalize_data_array():
    da = xr.DataArray([10, 20])
    result = OmnesDataArray.normalize_data(da)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, [10, 20])


def test_accessor_attached(base_oda):
    accessor = base_oda.omnes
    assert isinstance(accessor, OmnesAccessor)
