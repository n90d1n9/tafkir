package tech.kayys.tafkir.train.diffusion.api;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Applies a backend-specific optimization step for a computed loss tensor.
 */
@FunctionalInterface
public interface DiffusionOptimizationStep {

    void update(Tensor loss);
}
