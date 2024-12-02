from os import makedirs
from os.path import join

from pandas import DataFrame

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from input.definitions import DataKind
from utility import configuration
from utility.enum_definitions import convert_enum_to_value


class Write(PipelineStage):
    _name = "output_writer"
    stage = Stage.WRITE_OUT
    csv_properties = {"sep": ';', "index": True, "float_format": '%.4f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "output")
        self.filename = kwargs.get("filename", name)
        makedirs(self.output_path, exist_ok=True)

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        name = kwargs.get("filename", self.filename)
        if DataKind.MUNICIPALITY.value not in dataset.dims:
            self.write_array(dataset, name)
            return dataset

        for municipality in dataset[DataKind.MUNICIPALITY.value].values:
            makedirs(join(self.output_path, municipality), exist_ok=True)
            self.write_array(dataset.sel({DataKind.MUNICIPALITY.value: municipality}), name, municipality=municipality)
        return dataset

    def write_array(self, dataset: OmnesDataArray, name, **kwargs):
        self.save_2d_data_array(dataset, name, **kwargs)

    def save_2d_data_array(self, dataset, name, **kwargs):
        output = dataset.to_pandas().map(convert_enum_to_value).rename(columns=convert_enum_to_value,
                                                                       index=convert_enum_to_value)
        output.to_csv(join(self.output_path, kwargs.pop("municipality", ""), name if ".csv" in name else f"{name}.csv"),
                      **self.csv_properties, index_label=output.index.name, **kwargs)

    def write(self, output: DataFrame, name=None):
        self.execute(output.to_xarray(), filename=name)


class WriteGseProfile(Write):
    _name = "output_writer"
    csv_properties = {"sep": ';', "index": True, "float_format": '%.8f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "reference_profile_source")
        self.filename = kwargs.get("filename", "gse_ref_profiles")

    def execute(self, dataset: OmnesDataArray, *args, **kwargs) -> OmnesDataArray:
        name = kwargs.get("filename", self.filename)
        self.save_2d_data_array(dataset, name)
        return dataset


class WriteSeparately(Write):
    _name = "separated_writer"
    csv_properties = {"sep": ';', "index": True, "float_format": '%.8f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = join(self.output_path, kwargs.get("subdirectory"))
        makedirs(self.output_path, exist_ok=True)
        self.separate_by = kwargs.get("separate_by", "user")

    def write_array(self, dataset: OmnesDataArray, name, **kwargs):
        for idx, da in dataset.groupby(self.separate_by):
            self.save_2d_data_array(da, idx)
