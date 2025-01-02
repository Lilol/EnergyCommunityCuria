from collections import OrderedDict

from data_processing_pipeline.data_processing_arbiter import DataProcessingArbiter
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.dataset import OmnesDataArray


class TwoWayDictionary(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._key_map = OrderedDict()

    def set(self, key1, key2, value):
        combined_key = frozenset((key1, key2))
        self[key1] = value
        self[key2] = value
        self._key_map[combined_key] = value

    def get(self, key1=None, key2=None):
        # Get a value using one or two keys.
        if key1 is None and key2 is None:
            raise ValueError("Both keys cannot be None at the same time")

        combined_key = frozenset((key1, key2))
        return self._key_map.get(combined_key, super().get(key1))

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            self.set(*key, value)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self.get(*key)
        elif isinstance(key, str):
            return super().__getitem__(key)
        else:
            raise TypeError(f"'{self.__class__.__name__}' object is not subscriptable")

    def __delitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            del self[key[0]]
            del self[key[1]]
            combined_key = frozenset(key)
            del self._key_map[combined_key]
        else:
            super().__delitem__(key)

    def __contains__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return key[0] in self and key[1] in self
        return super().__contains__(key)

    def __len__(self):
        return len(set(self._key_map.keys()))

    def __iter__(self):
        return iter(self._key_map.items())


class DataProcessingPipeline(TwoWayDictionary):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self._dataset = kwargs.pop("dataset", None)
        DataProcessingArbiter().register_pipeline(name, self)
        # Set pipeline workers
        # Pipeline workers' order is determined by their order of appearance
        for worker in kwargs.pop("workers", [PipelineStage(), ]):
            self.set(worker.stage, worker.name, worker)

    def register(self, worker: PipelineStage):
        self.set(worker.stage, worker.name, worker)

    def set_dataset(self, dataset):
        self._dataset = dataset

    def execute(self, *args, **kwargs) -> OmnesDataArray:
        dataset = kwargs.pop("dataset", self._dataset)
        for _, processor in iter(self):
            dataset = processor.execute(dataset, *args, **kwargs)
        return dataset
