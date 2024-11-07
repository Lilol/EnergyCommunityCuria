from enum import Enum
from os import makedirs
from os.path import join

from pandas import DataFrame

import configuration


class Writer:
    csv_properties = {"sep": ';', "index": False, "float_format": ".4f"}

    def __init__(self, *args, **kwargs):
        self.output_path = configuration.config("path", "output")
        makedirs(self.output_path, exist_ok=True)

    def write(self, output: DataFrame, name):
        output = output.map(lambda x: x.value if type(x)==Enum else x)
        output.to_csv(join(self.output_path, name if ".csv" not in name else f"{name}.csv"), **self.csv_properties)

