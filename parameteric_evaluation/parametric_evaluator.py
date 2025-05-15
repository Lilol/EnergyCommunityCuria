from data_storage.dataset import OmnesDataArray
from io_operation.input.definitions import DataKind
from parameteric_evaluation.calculator import Calculator
from parameteric_evaluation.definitions import ParametricEvaluationType, PhysicalMetric, LoadMatchingMetric, \
    EconomicMetric, EnvironmentalMetric
from utility import configuration
from utility.subclass_registration_base import SubclassRegistrationBase

"""Metaclass to initialize _parameter_calculators before registration."""


class EvaluatorMeta(type):
    @staticmethod
    def get_eval_metrics(evaluation_type):
        return {key: Calculator.get_subclass(key) for key in {ParametricEvaluationType.PHYSICAL_METRICS: PhysicalMetric,
                                                              ParametricEvaluationType.LOAD_MATCHING_METRICS: LoadMatchingMetric,
                                                              ParametricEvaluationType.ECONOMIC_METRICS: EconomicMetric,
                                                              ParametricEvaluationType.ENVIRONMENTAL_METRICS: EnvironmentalMetric, }.get(
            evaluation_type, [])}

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Ensure _key is set before using it
        if hasattr(cls, "_key") and cls._key is not None:
            cls._parameter_calculators = cls.get_eval_metrics(cls._key)


class ParametricEvaluator(SubclassRegistrationBase, metaclass=EvaluatorMeta):
    _key = ParametricEvaluationType.INVALID
    _name = "parametric_evaluator"
    _parameter_calculators = {}

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        results = kwargs.pop("results", args[0])
        dataset = kwargs.pop('dataset', args[1])
        battery_size = kwargs.get('battery_size')
        number_of_families = kwargs.get('number_of_families')
        pv_sizes = kwargs.get('pv_sizes')
        for metric, calculator in cls._parameter_calculators.items():
            if metric == EconomicMetric.INVALID:
                continue
            results=results.update(calculator.calculate(battery_size=battery_size, number_of_families=number_of_families,
                                                pv_sizes=pv_sizes), **{DataKind.BATTERY_SIZE.value: battery_size,
                                                                     DataKind.NUMBER_OF_FAMILIES.value: number_of_families,
                                                                     DataKind.METRIC.value: metric})
        return dataset

    @classmethod
    def create(cls, *args, **kwargs):
        evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")
        return {kind: cls._subclasses[kind](*args, **kwargs) for kind, evaluator_initialization_function in
                cls._subclasses.items() if kind in evaluation_types}
