package tech.kayys.tafkir.train.diffusion.api;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Java-first denoiser contract used by diffusion training algorithms.
 */
@FunctionalInterface
public interface DiffusionDenoiser {

    Tensor predict(Tensor latents, Tensor conditioning, int timestep);
}
