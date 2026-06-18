package tech.kayys.tafkir.ml.reasoning;

import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Target-side conditioning bundle for GRAM posterior training.
 */
public record GramTrainingTarget(
        Tensor targetEmbedding,
        Map<String, Object> metadata) {

    public GramTrainingTarget {
        targetEmbedding = Objects.requireNonNull(targetEmbedding, "targetEmbedding must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
