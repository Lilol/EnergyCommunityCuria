from utility.singleton import Singleton


class DataProcessingArbiter(Singleton):
    def __init__(self):
        self._pipelines = {}

    def register_pipeline(self, name, pipeline):
        self.__setitem__(name, pipeline)

    def __getitem__(self, item):
        return self._pipelines.get(item)

    def __setitem__(self, key, value):
        self._pipelines[key] = value
