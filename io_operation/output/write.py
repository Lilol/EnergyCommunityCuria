from os import makedirs
from os.path import join

from pandas import DataFrame

from data_processing_pipeline.definitions import Stage
from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.io_operation_separately_by_attribute import IoOperationSeparately
from utility import configuration
from utility.definitions import append_extension
from utility.enum_definitions import convert_enum_to_value


class Write(IoOperationSeparately):
    _name = "output_writer"
    stage = Stage.WRITE_OUT
    csv_properties = {"sep": ';', "index": True, "float_format": '%.4f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "output")
        self.filename = kwargs.get("filename", name)
        makedirs(self.output_path, exist_ok=True)

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        return super().execute(dataset, *args, **kwargs, separate_to_directories_by=DataKind.MUNICIPALITY.value)

    def _io_operation(self, dataset: OmnesDataArray | None, attribute=None, attribute_value=None, *args,
                      **kwargs) -> OmnesDataArray | None:
        makedirs(join(self.output_path, f"{attribute_value}"), exist_ok=True)
        self.save_data(dataset.sel({attribute: attribute_value}), **kwargs)
        return dataset

    def save_data(self, dataset, **kwargs):
        raise NotImplementedError("'save_data' must be implemented in subclass")


class WriteDataArray(Write):
    _name = "data_array_writer"

    def save_data(self, dataset: OmnesDataArray, **kwargs):
        filename = kwargs.get("filename", self.filename)
        output_path = join(self.output_path, kwargs.pop("attribute", ""), kwargs.pop("attribute_value", ""))
        makedirs(output_path, exist_ok=True)
        dataset.to_netcdf(join(output_path, append_extension(filename, '.nc')))


class Write2DData(Write):
    _name = "output_writer_2d"

    def save_data(self, dataset: OmnesDataArray, **kwargs):
        self.write(dataset.to_pandas(), **kwargs)

    def write(self, output: DataFrame, **kwargs):
        output_path = join(self.output_path, kwargs.pop("attribute", ""), kwargs.pop("attribute_value", ""))
        makedirs(output_path, exist_ok=True)
        filename = kwargs.get("filename", self.filename)
        output = output.map(convert_enum_to_value).rename(columns=convert_enum_to_value,
                                                                       index=convert_enum_to_value)
        output.to_csv(join(output_path, append_extension(filename, '.csv')), **self.csv_properties,
                      index_label=output.index.name, **kwargs)


class WriteGseProfile(Write2DData):
    _name = "output_writer"
    csv_properties = {"sep": ';', "index": True, "float_format": '%.8f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "reference_profile_source")
        self.filename = kwargs.get("filename", "gse_ref_profiles")

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        self.save_data(dataset, **kwargs)
        return dataset


class WriteSeparatelyToSubdir(Write2DData):
    _name = "separated_writer"
    csv_properties = {"sep": ';', "index": True, "float_format": '%.8f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.subdirectory = kwargs.get("subdirectory")
        self.separate_by = kwargs.get("separate_by", "user")

    def write_array(self, dataset: OmnesDataArray, name, **kwargs):
        for idx, da in dataset.groupby(self.separate_by):
            self.save_data(da, filename=idx, subdirectory=self.subdirectory, **kwargs)
