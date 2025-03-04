
from parameteric_evaluation.definitions import ParametricEvaluationType
from utility import configuration
from utility.singleton import Singleton


class ParametricEvaluator:
    _type = ParametricEvaluationType.INVALID
    _evaluators = {}

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
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
            evaluator.invoke(args, kwargs)


if __name__ == '__main__':
    ParametricEvaluation().run_evaluation()
