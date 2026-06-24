package tech.kayys.tafkir.trainer.optim;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Base interface for all Tafkir optimizers.
 *
 * <p>Implementations must update parameters in-place via
 * {@link TafkirTensor}'s in-place operations (e.g. {@code sub_}, {@code mul_}).
 *
 * <h3>Example</h3>
 * <pre>{@code
 * TafkirOptimizer opt = new TafkirAdam(params, 1e-3f);
 * // training loop
 * opt.step(params);
 * opt.zeroGrad(params);
 * // change LR on the fly
 * opt.setLearningRate(1e-4f);
 * }</pre>
 */
public interface TafkirOptimizer {

    /**
     * Performs one optimization step, updating all parameters in-place.
     *
     * @param parameters list of model parameter tensors
     */
    void step(List<TafkirTensor> parameters);

    /**
     * Zeroes the gradients of all parameters.
     *
     * @param parameters list of model parameter tensors
     */
    void zeroGrad(List<TafkirTensor> parameters);

    /**
     * Returns the current learning rate.
     *
     * @return learning rate
     */
    float getLearningRate();

    /**
     * Sets a new learning rate (used by LR schedulers).
     *
     * @param lr new learning rate
     */
    void setLearningRate(float lr);
}
