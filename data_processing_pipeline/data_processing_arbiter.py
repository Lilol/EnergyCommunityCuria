class DataProcessingArbiter(object):
    _instance = None

    def __init__(self):
        self._pipelines = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataProcessingArbiter, cls).__new__(cls)
        return cls._instance

    def register_pipeline(self, name, pipeline):
        self.__setitem__(name, pipeline)

    def __getitem__(self, item):
        return self._pipelines[item]

    def __setitem__(self, key, value):
        self._pipelines[key] = value
