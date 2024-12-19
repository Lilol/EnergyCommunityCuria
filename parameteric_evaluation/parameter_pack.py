import itertools
import logging
from collections import defaultdict

from utility import configuration

logger = logging.getLogger(__name__)


class EvaluationParameterPack:
    pattern = r'(\w+):?\s*(\[[^\]]*\])?(,\s*|\s*$)'

    @staticmethod
    def convert_to_int_vector(values):
        return [int(val) for val in values]

    @staticmethod
    def collect_combinations_from_non_complete_pairing(parameters):
        def return_key_set(container):
            return set(container.keys()) if isinstance(container, dict) else set(container)

        def return_value_set(container):
            return set(**container.values()) if isinstance(container, dict) else set(container)

        data = {"number_of_families": set(), "bess_sizes": set()}
        combinations = []
        if "number_of_families" in parameters and "bess_sizes" in parameters:
            data = parameters
            combinations = list(itertools.product(data["number_of_families"], data["bess_sizes"]))
        else:
            assert len(parameters) == 1
            key0 = next(iter(parameters))
            key1 = "number_of_families" if key0 == "bess_sizes" else "bess_sizes"
            for key, items in parameters[key0].items():
                d = defaultdict(set)
                values = return_value_set(items)
                d[key0].add(key)
                d[key1] = values
                data[key0].add(key)
                data[key1] = data[key1].union(values)
                combinations.append(list(itertools.product(d["number_of_families"], d["bess_sizes"])))
        return data["bess_sizes"], data["number_of_families"], combinations

    def __init__(self, parameters: str = None):
        if parameters is None:
            parameters = configuration.config.getstr("parametric_evaluation", "evaluation_parameters")
        self.parameters = eval(parameters)
        self.bess_sizes, self.number_of_families, self.combinations = self.collect_combinations_from_non_complete_pairing(
            self.parameters)

    def __iter__(self):
        return iter(self.combinations)
