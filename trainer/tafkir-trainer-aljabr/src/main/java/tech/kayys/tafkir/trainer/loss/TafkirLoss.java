package tech.kayys.tafkir.trainer.loss;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

/**
 * A loss function for training.
 */
public interface TafkirLoss {
    TafkirTensor compute(TafkirTensor pred, TafkirTensor target);
}
