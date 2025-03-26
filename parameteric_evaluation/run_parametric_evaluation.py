from parameteric_evaluation import initialize_evaluators
from parameteric_evaluation.dataset_creation import DatasetCreatorForParametricEvaluation
from parameteric_evaluation.metric_evaluator import MetricEvaluator
from utility.configuration import config

initialize_evaluators()


def run_evaluation():
    DatasetCreatorForParametricEvaluation.create_dataset_for_parametric_evaluation()
    MetricEvaluator.calculate_metrics(parameters=config.get("parametric_evaluation", "evaluation_parameters"))


if __name__ == '__main__':
    run_evaluation()
