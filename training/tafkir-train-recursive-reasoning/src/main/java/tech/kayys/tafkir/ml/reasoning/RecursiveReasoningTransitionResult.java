package tech.kayys.tafkir.ml.reasoning;

/**
 * Result of one stochastic latent transition in a recursive reasoning rollout.
 */
public record RecursiveReasoningTransitionResult(
        RecursiveReasoningState nextState,
        double logProbability,
        Double rewardScore) {
}
