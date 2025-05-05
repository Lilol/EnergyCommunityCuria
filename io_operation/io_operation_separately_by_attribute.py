from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray
from utility.definitions import get_value


class IoOperationSeparately(PipelineStage):
    _name = "io_operation_separately"
    stage = Stage.IO

    def io_operation(self, dataset: OmnesDataArray | None, attribute=None, attribute_value=None, *args,
                     **kwargs) -> OmnesDataArray | None:
        raise NotImplementedError("io_operation() must be implemented in all child classes.")

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        attribute = get_value(kwargs.get("separate_to_directories_by"))
        if attribute not in dataset.dims:
            return self.io_operation(dataset)

        for attribute_value in dataset[attribute].values:
            dataset = self.io_operation(dataset, attribute=attribute, attribute_value=attribute_value)
        return dataset

