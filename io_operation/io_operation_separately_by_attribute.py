from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.omnes_data_array import OmnesDataArray
from utility.definitions import get_value


class IoOperationSeparately(PipelineStage):
    _name = "io_operation_separately"
    stage = Stage.IO

    def _io_operation(self, dataset: OmnesDataArray | None, attribute=None, attribute_value=None, *args,
                     **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError("Function '_io_operation()' must be implemented in all child classes.")

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        attribute = get_value(self.get_arg("separate_to_directories_by", **kwargs))
        do_not_separate = self.get_arg("do_not_separate", fallback=False, **kwargs)
        if do_not_separate or "directories" not in kwargs and (dataset is None or attribute not in dataset.dims):
            return self._io_operation(dataset)

        attribute_values = self.get_arg("directories", **kwargs, fallback=None)
        if attribute_values is None:
            attribute_values = dataset[attribute].values
        for attribute_value in attribute_values:
            dataset = self._io_operation(dataset, attribute=attribute, attribute_value=attribute_value)
        return dataset
