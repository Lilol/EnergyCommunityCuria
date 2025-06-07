import logging
from collections import defaultdict

from data_storage.omnes_data_array import OmnesDataArray
from utility.singleton import Singleton

logger = logging.getLogger(__name__)


class DataStore(Singleton):
    # Data store: main object storing all dataarrays for processing
    _data = defaultdict(OmnesDataArray)

    def __getitem__(self, item):
        return self._data[item]

    def __delitem__(self, item):
        del self._data[item]

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
