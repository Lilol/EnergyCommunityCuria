from enum import Enum
from os import makedirs
from os.path import join

import xarray as xr
from pandas import DataFrame

from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from utility import configuration


class Writer(PipelineStage):
    csv_properties = {"sep": ';', "index": False, "float_format": ".4f"}

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config("path", "output")
        self.filename = kwargs.get("filename", None)
        makedirs(self.output_path, exist_ok=True)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs):
        name = kwargs.get("filename", self.filename)
        output = dataset.map(lambda x: x.value if type(x) == Enum else x)
        xr.apply_ufunc(lambda x: x.value if type(x) == Enum else x, dataset, vectorize=True)
        output.to_csv(join(self.output_path, name if ".csv" not in name else f"{name}.csv"), **self.csv_properties)

    def write(self, output: DataFrame, name=None):
        self.execute(output.to_xarray(), filename=name)
