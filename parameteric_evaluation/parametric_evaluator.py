from data_storage.dataset import OmnesDataArray
from parameteric_evaluation.definitions import ParametricEvaluationType
from utility import configuration
from utility.subclass_registration_base import SubclassRegistrationBase


class ParametricEvaluator(SubclassRegistrationBase):
    _type = ParametricEvaluationType.INVALID
    _name = "parametric_evaluator"
    _parameter_calculators = {}

    @classmethod
    def invoke(cls, *args, **kwargs) -> OmnesDataArray | float | None:
        results = kwargs.pop("results", args[0])
        dataset = kwargs.pop('dataset', args[1])
        for parameter, calculator in cls._parameter_calculators.items():
            results.loc[results.index[-1], parameter.to_abbrev_str()] = calculator.calculate(dataset, dataset)
        return dataset

    @classmethod
    def create(cls, *args, **kwargs):
        evaluation_types = configuration.config.get("parametric_evaluation", "to_evaluate")
        return {kind: cls._subclasses[kind](*args, **kwargs) for kind, evaluator_initialization_function in
                cls._subclasses.items() if kind in evaluation_types}

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     if len(cls._evaluators) == 0:
    #         cls._evaluators = {}
    #     cls._evaluators[cls._type] = cls
