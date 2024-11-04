from data_processing_pipeline.data_processing_arbiter import DataProcessingArbiter
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.data_store import DataStore
from data_storage.dataset import OmnesDataArray


class DataProcessingPipeline:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        DataProcessingArbiter().register_pipeline(name, self)
        self.workers = {worker.stage: worker for worker in kwargs.pop("workers", [PipelineStage(), ])}

    def register(self, worker: PipelineStage):
        self.workers[worker.stage] = worker

    def execute(self, dataset, *args, **kwargs) -> OmnesDataArray:
        for _, processor in self.workers.items():
            dataset = processor.execute(dataset, *args, **kwargs)
        DataStore()[self.name] = dataset
        return dataset
