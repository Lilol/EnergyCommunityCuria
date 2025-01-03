from parameteric_evaluation.dataset_creation import create_dataset_for_parametric_evaluation
from parameteric_evaluation.definitions import ParametricEvaluation

if __name__ == '__main__':
    # Evaluate the number of families to reach set targets
    create_dataset_for_parametric_evaluation()
    ParametricEvaluation().run_evaluation()
