package tech.kayys.tafkir.ml.gguf;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.Map;

/**
 * Represents a GGUF model containing tensors and metadata.
 */
public final class GgufModel {
    private final Map<String, TafkirTensor> tensors;
    private final Map<String, GgufMetaValue> metadata;

    public GgufModel(Map<String, TafkirTensor> tensors, Map<String, GgufMetaValue> metadata) {
        this.tensors = Map.copyOf(tensors);
        this.metadata = Map.copyOf(metadata);
    }

    public Map<String, TafkirTensor> getTensors() {
        return tensors;
    }

    public Map<String, GgufMetaValue> getMetadata() {
        return metadata;
    }

    public TafkirTensor getTensor(String name) {
        return tensors.get(name);
    }

    public GgufMetaValue getMetadata(String key) {
        return metadata.get(key);
    }
}