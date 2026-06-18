package tech.kayys.tafkir.ml.reasoning;

/**
 * Reduction used when aggregating deep-supervision losses across GRAM steps.
 */
public enum GramObjectiveReduction {
    SUM,
    MEAN
}
