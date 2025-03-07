from data_storage.dataset import OmnesDataArray
from parameteric_evaluation.definitions import ParametricEvaluationType, get_eval_metrics
from utility import configuration
from utility.subclass_registration_base import SubclassRegistrationBase


class EvaluatorMeta(type):
    """Metaclass to initialize _parameter_calculators before registration."""

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Ensure _key is set before using it
        if hasattr(cls, "_key") and cls._key is not None:
            cls._parameter_calculators = get_eval_metrics(cls._key)


class ParametricEvaluator(SubclassRegistrationBase, metaclass=EvaluatorMeta):
    _key = ParametricEvaluationType.INVALID
    _name = "parametric_evaluator"
    _parameter_calculators = {}

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        results = kwargs.pop("results", args[0])
        dataset = kwargs.pop('dataset', args[1])
        for parameter, calculator in cls._subclasses.items():
            results.loc[results.index[-1], parameter.to_abbrev_str()] = calculator.calculate(dataset, dataset)
        return dataset

    @classmethod
    def create(cls, *args, **kwargs):
        evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")
        return {kind: cls._subclasses[kind](*args, **kwargs) for kind, evaluator_initialization_function in
                cls._subclasses.items() if kind in evaluation_types}
