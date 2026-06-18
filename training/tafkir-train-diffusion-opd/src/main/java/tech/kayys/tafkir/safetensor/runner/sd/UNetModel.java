package tech.kayys.aljabr.safetensor.runner.sd;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;

/**
 * Stub — UNet model from the safetensor Stable Diffusion runner.
 *
 * <p>The real implementation is in the safetensor runner module and binds to
 * native GGUF / safetensor weights. This stub unblocks compilation of the adapter layer.
 */
public abstract class UNetModel implements AutoCloseable {

    /**
     * Run a forward pass of the UNet.
     *
     * @param latents      noisy latent input
     * @param timestep     diffusion timestep
     * @param conditioning text or image conditioning tensor
     * @return predicted noise (caller must close)
     */
    public abstract AccelTensor predict(AccelTensor latents, int timestep, AccelTensor conditioning);

    @Override
    public void close() {
        // Stub: no native resources
    }
}
