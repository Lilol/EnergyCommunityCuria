import json

from utility import configuration


class EvaluationParameterPack:
    def __init__(self, parameters: str=None):
        if parameters is None:
            parameters = configuration.config.getstr("parametric_evaluation", "to_evaluate")
        self.parameters = json.loads(parameters)
        self.number_of_families = self.parameters["number_of_families"]
        self.bess_sizes = self.parameters["bess_sizes"]
