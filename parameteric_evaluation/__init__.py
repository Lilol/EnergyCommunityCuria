from parameteric_evaluation.parametric_evaluator import ParametricEvaluator


# Function to trigger loading only when needed
def initialize_evaluators():
    from parameteric_evaluation.economic import EconomicEvaluator
    from parameteric_evaluation.environmental import EnvironmentalEvaluator
    from parameteric_evaluation.physical import PhysicalMetricEvaluator
    from parameteric_evaluation.load_matching_evaluation import LoadMatchingMetricEvaluator
    from parameteric_evaluation.target_self_consumption import TargetSelfConsumptionEvaluator
    from parameteric_evaluation.time_aggregation_evaluation import TimeAggregationEvaluator

