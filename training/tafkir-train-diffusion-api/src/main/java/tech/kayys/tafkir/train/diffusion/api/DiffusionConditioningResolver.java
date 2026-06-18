package tech.kayys.tafkir.train.diffusion.api;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Resolves model conditioning tensors for a prompt sample.
 */
@FunctionalInterface
public interface DiffusionConditioningResolver {

    Tensor resolve(DiffusionPromptSample sample);
}
