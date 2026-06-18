package tech.kayys.aljabr.diffusion.model;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * SPI stub — Java-native UNet model contract.
 * Real implementations live in runner modules.
 */
public interface UNetModel {
    /**
     * Run a forward pass through the UNet.
     *
     * @param latents      the noisy latent tensor
     * @param conditioning the text/image conditioning tensor
     * @param timestep     the diffusion timestep
     * @return predicted noise tensor
     */
    Tensor predict(Tensor latents, Tensor conditioning, int timestep);
}
