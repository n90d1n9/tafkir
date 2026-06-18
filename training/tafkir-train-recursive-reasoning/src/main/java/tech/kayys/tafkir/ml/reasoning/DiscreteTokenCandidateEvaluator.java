package tech.kayys.tafkir.ml.reasoning;

/**
 * Scores one discrete token prediction for a benchmark-specific task.
 */
@FunctionalInterface
public interface DiscreteTokenCandidateEvaluator {
    DiscreteTokenEvaluation evaluate(DiscreteTrajectoryTokenPrediction prediction);
}
