package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * A single layer in a Tafkir model.
 */
public interface TafkirLayer {
    TafkirTensor forward(TafkirTensor input, boolean training);
    List<TafkirTensor> parameters();
    default long parameterCount() {
        return parameters().stream().mapToLong(TafkirTensor::numel).sum();
    }
}
