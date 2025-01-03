from enum import auto

from parameteric_evaluation.dataset_creation import DatasetCreatorForParametricEvaluation
from utility import configuration
from utility.definitions import OrderedEnum
from utility.singleton import Singleton


class ParametricEvaluationType(OrderedEnum):
    DATASET_CREATION = "dataset_creation"
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
    INJECTED_ENERGY = "Injected energy"
    WITHDRAWN_ENERGY = "Withdrawn energy"
    ESR = "Emissions savings ratio"
    TOTAL_EMISSIONS = "Total emissions"
    BASELINE_EMISSIONS = "Baseline emissions"
    CAPEX = "Capex"
    OPEX = "Opex"
    INVALID = auto()


class ParametricEvaluator:
    _type = ParametricEvaluationType.INVALID
    _evaluators = {ParametricEvaluationType.DATASET_CREATION: DatasetCreatorForParametricEvaluation}

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self):
        raise NotImplementedError("The 'invoke' method must be implemented in all derived classes.")

    @classmethod
    def create(cls, *args, **kwargs):
        evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")
        return {kind: cls._evaluators[kind](*args, **kwargs) for kind, evaluator_initialization_function in
                cls._evaluators.items() if kind in evaluation_types}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls._evaluators) == 0:
            cls._evaluators = {}
        cls._evaluators[cls._type] = cls


class ParametricEvaluation(Singleton):
    _evaluators = ParametricEvaluator.create()

    def run_evaluation(self, *args, **kwargs):
        for name, evaluator in self._evaluators.items():
            evaluator.run(args, kwargs)
