import functools
from enum import Enum
from typing import Iterable

import numpy as np
from pandas import MultiIndex
from xarray import DataArray

from input.definitions import DataKind
from utility import configuration


def append_extension(name, ext='.csv'):
    return name if ext in name else f"{name}{ext}"


@functools.total_ordering
class OrderedEnum(Enum):
    @classmethod
    @functools.lru_cache(None)
    def _member_list(cls):
        return list(cls)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            member_list = self.__class__._member_list()
            return member_list.index(self) < member_list.index(other)
        return NotImplemented


def grouper(data_array, *grouper_dims, **grouper_vars):
    def check_grouper_vars(**gv):
        for key, val in gv.items():
            if not isinstance(val, Iterable):
                gv[key] = [val, ]
        return gv

    grouper_vars = check_grouper_vars(**grouper_vars)
    if grouper_dims:
        grouper_coords = {grouper_dims[0]: data_array[grouper_dims[0]].values}
    else:
        gr_coord = list(grouper_vars.keys())[0]
        dims = np.array(data_array.dims)
        dim = dims[dims != gr_coord][0]
        grouper_coords = {dim: data_array[dim].values}
    return DataArray(MultiIndex.from_arrays((*[data_array[gc].values for gc in grouper_dims],
                                             *[data_array.sel({key: value}).squeeze().values for key, values in
                                               grouper_vars.items() for value in values]),
                                            names=list(grouper_dims) + list(*grouper_vars.values())),
                     dims=list(grouper_coords.keys()), coords=grouper_coords, )
