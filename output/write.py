from enum import Enum
from os import makedirs
from os.path import join

from pandas import DataFrame

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import ColumnName
from utility import configuration


class Write(PipelineStage):
    _name = "output_writer"
    stage = Stage.WRITE_OUT
    csv_properties = {"sep": ';', "index": True, "float_format": '%.4f'}

    @staticmethod
    def convert_enum_to_value(x):
        return x.value if isinstance(x, Enum) else x

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "output")
        self.filename = kwargs.get("filename", name)
        makedirs(self.output_path, exist_ok=True)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        name = kwargs.get("filename", self.filename)
        if ColumnName.MUNICIPALITY.value not in dataset.dims:
            self.save_2d_dataarray(dataset, name)
            return dataset

        for municipality in dataset[ColumnName.MUNICIPALITY.value].values:
            makedirs(join(self.output_path, municipality), exist_ok=True)
            self.save_2d_dataarray(dataset.sel({ColumnName.MUNICIPALITY.value: municipality}), name,
                                   municipality=municipality)
        return dataset

    def save_2d_dataarray(self, dataset, name, **kwargs):
        output = dataset.to_pandas().map(self.convert_enum_to_value).rename(columns=self.convert_enum_to_value,
                                                                            index=self.convert_enum_to_value)
        output.to_csv(join(self.output_path, kwargs.pop("municipality", ""), name if ".csv" in name else f"{name}.csv"),
                      **self.csv_properties, index_label=output.index.name)

    def write(self, output: DataFrame, name=None):
        self.execute(output.to_xarray(), filename=name)
