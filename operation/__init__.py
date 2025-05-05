from operation.profile_scaling import ScaleProfile

def initialize_operation():
    from operation.quadratic_optimization_scaler import ScaleByQuadraticOptimization
    from operation.linear_equation_scaler import ScaleByLinearEquation
    from operation.time_of_use_scaler import ScaleTimeOfUseProfile
    from operation.proportional_scaler import ScaleInProportion
    from operation.flat_scaler import ScaleFlat
