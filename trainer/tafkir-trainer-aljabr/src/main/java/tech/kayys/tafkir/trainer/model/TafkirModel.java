package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * A trainable model in Tafkir.
 * Models must support both forward pass and gradient computation.
 */
public interface TafkirModel {
    /**
     * Forward pass. Must support gradient computation (requiresGrad=true on parameters).
     */
    TafkirTensor forward(TafkirTensor input);

    /**
     * Returns all trainable parameters.
     */
    List<TafkirTensor> parameters();

    /**
     * Sets training mode (affects dropout, batch norm, etc.)
     */
    void train();

    /**
     * Sets evaluation mode.
     */
    void eval();

    /**
     * Returns true if the model is in training mode.
     */
    boolean isTraining();
}
