import json
from json import JSONDecodeError

from utility import configuration


class EvaluationParameterPack:
    @staticmethod
    def convert_to_int_vector(values):
        return [int(val) for val in values]

    @staticmethod
    def collect_combinations_from_non_complete_pairing(parameters):
        values1 = set()
        values2 = set()
        combinations = []
        for key, items in parameters.items():
            k = int(key)
            values1.add(k)
            for v in items:
                val = int(v)
                values2.add(val)
                combinations.append((k, val))
        return values1, values2, combinations

    def __init__(self, parameters: str = None):
        if parameters is None:
            parameters = configuration.config.getstr("parametric_evaluation", "to_evaluate")
        try:
            self.parameters = json.loads(parameters).to_dict()
        except JSONDecodeError:
            self.parameters = {i.strip('{').split(': ')[0]: i.split(': ')[1].strip('}').strip('[]').split(',') for i in parameters.split(', ')}

        if "self_consumption_targets" in self.parameters:
            self.self_consumption_targets = self.convert_to_int_vector(self.parameters["self_consumption_targets"])

        self.combinations = []
        if "number_of_families" in self.parameters:
            try:
                self.number_of_families = self.convert_to_int_vector(self.parameters["number_of_families"])
                self.bess_sizes = self.convert_to_int_vector(self.parameters["bess_sizes"])
                self.combinations = [(f, b) for f in self.number_of_families for b in self.bess_sizes]
                return
            except ValueError:
                self.number_of_families, self.bess_sizes, self.combinations = self.collect_combinations_from_non_complete_pairing(
                    self.parameters["number_of_families"])
                return
        if "bess_sizes" in self.parameters:
            self.bess_sizes, self.number_of_families, self.combinations = self.collect_combinations_from_non_complete_pairing(
                    self.parameters["bess_sizes"])

    def __iter__(self):
        return iter(self.combinations)
