package tech.kayys.tafkir.ml.reasoning;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Deterministic proposal update before GRAM stochastic guidance is applied.
 */
@FunctionalInterface
public interface GramDeterministicTransition {
    Tensor propose(
            RecursiveReasoningState previousState,
            RecursiveReasoningContext context);
}
