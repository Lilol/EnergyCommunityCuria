import itertools
import logging
from collections import defaultdict

from io_operation.input.definitions import DataKind
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

        data = {DataKind.NUMBER_OF_FAMILIES.value: set(), DataKind.BATTERY_SIZE.value: set()}
        combinations = []
        if DataKind.NUMBER_OF_FAMILIES.value in parameters and DataKind.BATTERY_SIZE.value in parameters:
            data = parameters
            combinations = list(
                itertools.product(data[DataKind.NUMBER_OF_FAMILIES.value], data[DataKind.BATTERY_SIZE.value]))
        else:
            assert len(parameters) == 1
            key0 = next(iter(parameters))
            key1 = DataKind.NUMBER_OF_FAMILIES.value if key0 == DataKind.BATTERY_SIZE.value else DataKind.BATTERY_SIZE.value
            for key, items in parameters[key0].items():
                d = defaultdict(set)
                values = return_value_set(items)
                d[key0].add(key)
                d[key1] = values
                data[key0].add(key)
                data[key1] = data[key1].union(values)
                combinations.append(
                    list(itertools.product(d[DataKind.NUMBER_OF_FAMILIES.value], d[DataKind.BATTERY_SIZE.value])))
        return data[DataKind.BATTERY_SIZE.value], data[DataKind.NUMBER_OF_FAMILIES.value], combinations

    def __init__(self, parameters: str = None):
        if parameters is None:
            parameters = configuration.config.getstr("parametric_evaluation", "evaluation_parameters")
        self.parameters = eval(parameters)
        self.bess_sizes, self.number_of_families, self.combinations = self.collect_combinations_from_non_complete_pairing(
            self.parameters)

    def __iter__(self):
        for n_fam, bess_size in self.combinations:
            yield {
                "number_of_families": n_fam,
                "battery_size": bess_size
            }
