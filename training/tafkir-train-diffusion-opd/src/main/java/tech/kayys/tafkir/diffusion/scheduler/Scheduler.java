package tech.kayys.aljabr.diffusion.scheduler;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * SPI stub — Java-native diffusion scheduler contract.
 * Real implementations live in runner modules.
 */
public interface Scheduler {
    /**
     * Advance one denoising step.
     *
     * @param xT             the current noisy latent
     * @param modelPrediction the UNet's noise prediction
     * @param timestepIndex  index into the timestep schedule
     * @return the denoised latent after this step
     */
    Tensor step(Tensor xT, Tensor modelPrediction, int timestepIndex);

    /** Returns the full ordered array of timestep values. */
    int[] timesteps();
}
