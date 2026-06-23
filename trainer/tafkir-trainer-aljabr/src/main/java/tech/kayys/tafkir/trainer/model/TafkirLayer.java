package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * A single layer in a Tafkir model.
 */
public interface TafkirLayer {
    /**
     * Forward pass through this layer.
     *
     * @param input the input tensor
     * @param training true if in training mode (enables dropout, etc.)
     * @return the output tensor
     */
    TafkirTensor forward(TafkirTensor input, boolean training);

    /**
     * Returns all trainable parameters in this layer.
     */
    List<TafkirTensor> parameters();

    /**
     * Returns the number of parameters in this layer.
     */
    default long parameterCount() {
        return parameters().stream().mapToLong(TafkirTensor::numel).sum();
    }
}
