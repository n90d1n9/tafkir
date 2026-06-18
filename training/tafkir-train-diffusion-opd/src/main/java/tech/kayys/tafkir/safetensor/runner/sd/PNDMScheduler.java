package tech.kayys.aljabr.safetensor.runner.sd;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;

/**
 * Stub — PNDM scheduler from the safetensor Stable Diffusion runner.
 *
 * <p>The real implementation is in the safetensor runner module.
 * Provides the metadata and step interface used by {@code StableDiffusionRunnerAdapters}.
 */
public abstract class PNDMScheduler implements AutoCloseable {

    /** Scheduler family identifier (e.g. {@code "pndm"}). */
    public abstract String family();

    /** Number of denoising steps. */
    public abstract int stepCount();

    /** Number of training timesteps (e.g. 1000). */
    public abstract int trainingTimestepCount();

    /** Ordered array of timestep values for the current schedule. */
    public abstract int[] timestepsArray();

    /** Alpha cumprod values used internally by the PNDM algorithm. */
    public abstract float[] alphasCumprod();

    /**
     * Advance one denoising step.
     *
     * @param modelOutput   the UNet's noise prediction
     * @param timestep      the current timestep value
     * @param sample        the current noisy latent
     * @return the denoised latent after this step (caller must close)
     */
    public abstract AccelTensor step(AccelTensor modelOutput, int timestep, AccelTensor sample);

    @Override
    public void close() {
        // Stub: no native resources
    }
}
