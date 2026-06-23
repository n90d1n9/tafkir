package tech.kayys.tafkir.trainer.loss;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

/**
 * A loss function for training.
 */
public interface TafkirLoss {
    /**
     * Computes the loss between predictions and targets.
     *
     * @param pred model predictions (e.g., [batch, numClasses] log-probabilities)
     * @param target ground truth (e.g., [batch] class indices, or [batch, numClasses] one-hot)
     * @return scalar loss tensor
     */
    TafkirTensor compute(TafkirTensor pred, TafkirTensor target);
}
