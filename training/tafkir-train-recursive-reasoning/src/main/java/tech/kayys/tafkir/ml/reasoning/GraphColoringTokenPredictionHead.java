package tech.kayys.tafkir.ml.reasoning;

/**
 * Converts a GRAM transition sample into Graph Coloring prediction tokens.
 */
@FunctionalInterface
public interface GraphColoringTokenPredictionHead {
    GraphColoringTokenProjectionResult predictTokens(
            RecursiveReasoningState proposedState,
            RecursiveReasoningContext context,
            GramTransitionSample sample,
            GraphColoringProblem problem);
}
