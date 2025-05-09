from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from utility.definitions import get_value


class IoOperationSeparately(PipelineStage):
    _name = "io_operation_separately"
    stage = Stage.IO

    def _io_operation(self, dataset: OmnesDataArray | None, attribute=None, attribute_value=None, *args,
                     **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError("io_operation() must be implemented in all child classes.")

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        attribute = get_value(kwargs.get("separate_to_directories_by"))
        if "directories" not in kwargs and (dataset is None or attribute not in dataset.dims):
            return self._io_operation(dataset)

        attribute_values = kwargs.get("directories", None)
        if attribute_values is None:
            attribute_values = dataset[attribute].values
        for attribute_value in attribute_values:
            dataset = self._io_operation(dataset, attribute=attribute, attribute_value=attribute_value)
        return dataset

