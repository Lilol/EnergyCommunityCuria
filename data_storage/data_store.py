import logging
from collections import defaultdict

from data_storage.dataset import OmnesDataArray

logger = logging.getLogger(__name__)


class DataStore(object):
    # Data store: main object storing all dataarrays for processing
    _instance = None
    _data = defaultdict(OmnesDataArray)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if type(value) != OmnesDataArray:
            try:
                self._data[key] = OmnesDataArray(data=value)
            except ValueError as e:
                logger.error(f"Unexpected data type in DataStore.__setitem__, DataArray cannot be initialized.\n'{e}'")
                return
            else:
                return
        self._data[key] = value

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataStore, cls).__new__(cls)
        return cls._instance
