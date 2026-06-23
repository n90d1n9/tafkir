package tech.kayys.tafkir.trainer.optim;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Base interface for all optimizers.
 */
public interface TafkirOptimizer {
    void step(List<TafkirTensor> parameters);
    void zeroGrad(List<TafkirTensor> parameters);
}
