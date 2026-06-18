package tech.kayys.tafkir.ml.reasoning;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Provides epsilon noise for GRAM's reparameterized latent sampling.
 */
@FunctionalInterface
public interface GramNoiseSampler {
    Tensor sampleEpsilon(
            GramLatentGaussian distribution,
            GramTransitionInput input);
}
