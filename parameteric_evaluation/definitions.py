from enum import auto

from utility import configuration
from utility.definitions import OrderedEnum
from utility.singleton import Singleton


class ParametricEvaluationType(OrderedEnum):
    SELF_CONSUMPTION_TARGETS = "self_consumption_targets"
    TIME_AGGREGATION = "time_aggregations"
    METRICS = "metrics"
    PHYSICAL_METRICS = "physical"
    ECONOMIC_METRICS = "economic"
    ENVIRONMENTAL_METRICS = "environmental"
    ALL = "all"
    INVALID = "invalid"


class Parameter(OrderedEnum):
    SHARED_ENERGY = "Shared energy"
    SELF_CONSUMPTION = "Self consumption"
    SELF_SUFFICIENCY = "Self sufficiency"
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = auto()


class ParametricEvaluation(Singleton):
    _evaluators = {}

    def run_evaluation(self):
        for name, evaluator in self._evaluators.items():
            evaluator.evaluate()


class ParametricEvaluator:
    _type = ParametricEvaluationType.INVALID
    _evaluators = {}

    @classmethod
    def create(cls, *args, **kwargs):
        evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")
        if cls._type not in evaluation_types:
            return None
        return cls._evaluators[cls._type](*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls._evaluators) == 0:
            cls.stage_workers = {}
        cls.stage_workers[cls._type] = cls
