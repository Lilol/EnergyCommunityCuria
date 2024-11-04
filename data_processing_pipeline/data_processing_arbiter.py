class DataProcessingArbiter(object):
    _instance = None

    def __init__(self):
        self._pipelines = {}

    @classmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataProcessingArbiter, cls).__new__(cls)
        return cls._instance

    # TODO: should this work like a dictionary, or define separate methods for registering and retrieving pipelines?
    def register_pipeline(self, name, pipeline):
        self._pipelines[name] = pipeline

    def __getitem__(self, item):
        return self._pipelines[item]


