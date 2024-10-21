import logging
from collections import defaultdict

from xarray import DataArray

logger = logging.getLogger(__name__)


class DataStore(object):
    # Data store: main object storing all data for the operations
    _instance = None

    def __init__(self):
        self._data = defaultdict(DataArray)

    def __setitem__(self, key, value):
        if type(value) != DataArray:
            try:
                self._data[key] = DataArray(data=value)
            except ValueError as e:
                logger.error(f"Unexpected data type in DataStore.__setitem__, DataArray cannot be initialized. '{e}'")
                return
            else:
                return
        self._data[key] = value

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataStore, cls).__new__(cls)
        return cls._instance
