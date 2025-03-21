from pandas import DataFrame

from parameteric_evaluation import initialize_evaluators
from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from utility.singleton import Singleton

initialize_evaluators()


class EvaluationRunner(Singleton):
    _evaluators = ParametricEvaluator.create()

    def run_evaluation(self, *args, **kwargs):
        results = DataFrame()
        for name, evaluator in self._evaluators.items():
            evaluator.invoke(args, kwargs, results=results)


if __name__ == '__main__':
    EvaluationRunner().run_evaluation()
