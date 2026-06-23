package tech.kayys.tafkir.trainer.optim;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Base interface for all optimizers.
 */
public interface TafkirOptimizer {
    /**
     * Performs a single optimization step (parameter update).
     *
     * @param parameters the model parameters to update
     */
    void step(List<TafkirTensor> parameters);

    /**
     * Zeroes the gradients of all parameters.
     *
     * @param parameters the model parameters
     */
    void zeroGrad(List<TafkirTensor> parameters);
}
