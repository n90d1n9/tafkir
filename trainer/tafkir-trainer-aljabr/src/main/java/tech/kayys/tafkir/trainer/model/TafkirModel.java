package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * A trainable model in Tafkir.
 */
public interface TafkirModel {
    TafkirTensor forward(TafkirTensor input);
    List<TafkirTensor> parameters();
    void train();
    void eval();
    boolean isTraining();
}
