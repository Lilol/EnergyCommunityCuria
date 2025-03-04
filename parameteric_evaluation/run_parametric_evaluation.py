from parameteric_evaluation.parametric_evaluator import ParametricEvaluator
from utility.singleton import Singleton


class ParametricEvaluation(Singleton):
    _evaluators = ParametricEvaluator.create()

    def run_evaluation(self, *args, **kwargs):
        for name, evaluator in self._evaluators.items():
            evaluator.invoke(args, kwargs)


if __name__ == '__main__':
    ParametricEvaluation().run_evaluation()
