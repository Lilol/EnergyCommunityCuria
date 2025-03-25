from parameteric_evaluation import initialize_evaluators
from parameteric_evaluation.metric_evaluator import MetricEvaluator
from utility.singleton import Singleton

initialize_evaluators()


class EvaluationRunner(Singleton):
    @classmethod
    def run_evaluation(cls, **kwargs):
        MetricEvaluator.calculate_metrics(parameters=kwargs["parameters"])


if __name__ == '__main__':
    EvaluationRunner().run_evaluation()
