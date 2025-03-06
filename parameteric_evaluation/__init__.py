from parameteric_evaluation.parametric_evaluator import ParametricEvaluator

# Function to trigger loading only when needed
def initialize_evaluators():
    import parameteric_evaluation.metrics  # Triggers subclass definition & registration
    import parameteric_evaluation.economic
    import parameteric_evaluation.environmental
    import parameteric_evaluation.physical