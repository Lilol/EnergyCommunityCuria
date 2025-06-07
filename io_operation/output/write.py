import logging
from os import makedirs
from os.path import join

from pandas import DataFrame

from data_processing_pipeline.definitions import Stage
from data_storage.omnes_data_array import OmnesDataArray
from io_operation.input.definitions import DataKind
from io_operation.io_operation_separately_by_attribute import IoOperationSeparately
from utility import configuration
from utility.definitions import append_extension
from utility.enum_definitions import convert_enum_to_value

logger = logging.getLogger(__name__)


class Write(IoOperationSeparately):
    _name = "output_writer"
    stage = Stage.WRITE_OUT
    csv_properties = {"sep": ';', "index": True, "float_format": '%.4f'}

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_path = configuration.config.get("path", "output")
        self.filename = kwargs.get("filename", name)
        makedirs(self.output_path, exist_ok=True)

    def execute(self, dataset: OmnesDataArray | None, separate_to_directories_by=DataKind.MUNICIPALITY.value, *args,
                **kwargs) -> OmnesDataArray | None:
        return super().execute(dataset, *args, **kwargs, separate_to_directories_by=separate_to_directories_by)

    def _io_operation(self, dataset: OmnesDataArray | None, attribute=None, attribute_value=None, *args,
                      **kwargs) -> OmnesDataArray | None:
        makedirs(join(self.output_path, f"{attribute_value}"), exist_ok=True)
        self.save_data(dataset.sel({attribute: attribute_value}) if attribute is not None else dataset,
                       attribute=attribute,
                       attribute_value=attribute_value if attribute is not None else "",
                       **kwargs)
        return dataset

    def save_data(self, dataset, **kwargs):
        raise NotImplementedError("'save_data' must be implemented in subclass")


class WriteDataArray(Write):
    _name = "data_array_writer"

    def save_data(self, dataset: OmnesDataArray, **kwargs):
        filename = self.get_arg("filename", **kwargs, fallback=self.filename)
        attribute_value = self.get_arg("attribute_value", **kwargs, fallback="")
        output_path = join(self.output_path, attribute_value)
        makedirs(output_path, exist_ok=True)
        dataset = dataset.assign_coords(
            {dim: [convert_enum_to_value(coord) for coord in dataset[dim].values] for dim in dataset.dims})
        filename = join(output_path, append_extension(filename, '.nc'))
        try:
            dataset.to_netcdf(filename)
        except ValueError:
            logger.warning(
                f"Writing file '{filename}' failed due to dataarray containing mixed types, retrying with values converted to string")
            dataset.astype(str).to_netcdf(filename)
            logger.info(f"Writing file '{filename}' was written successfully")


class Write2DData(Write):
    _name = "output_writer_2d"

    def save_data(self, dataset: OmnesDataArray, **kwargs):
        self.write(dataset.to_pandas(), **kwargs)

    def write(self, output: DataFrame, **kwargs):
        output_path = join(self.output_path, kwargs.pop("attribute", ""), kwargs.pop("attribute_value", ""))
        makedirs(output_path, exist_ok=True)
        filename = self.get_arg("filename", **kwargs, fallback=self.filename)
        output = output.map(convert_enum_to_value).rename(columns=convert_enum_to_value, index=convert_enum_to_value)
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
