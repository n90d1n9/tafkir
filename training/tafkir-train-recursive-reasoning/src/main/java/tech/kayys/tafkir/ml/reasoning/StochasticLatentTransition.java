package tech.kayys.tafkir.ml.reasoning;

/**
 * Shared contract for GRAM-style stochastic latent transitions.
 */
@FunctionalInterface
public interface StochasticLatentTransition {

    RecursiveReasoningTransitionResult sample(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context);
}
