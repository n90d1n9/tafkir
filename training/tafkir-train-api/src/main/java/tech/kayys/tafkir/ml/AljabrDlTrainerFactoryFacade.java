package tech.kayys.tafkir.ml;

import tech.kayys.tafkir.ml.autograd.Acceleration;
import tech.kayys.tafkir.ml.train.CanonicalTrainer;

/**
 * Trainer and runtime construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlTrainerFactoryFacade extends AljabrDlLossFactoryFacade {
    protected AljabrDlTrainerFactoryFacade() {
    }

    public static CanonicalTrainer.Builder trainer() {
        return CanonicalTrainer.builder();
    }

    public static tech.kayys.tafkir.train.diffusion.opd.DefaultDiffusionOpdTrainer.Builder diffusionOpdTrainer() {
        return tech.kayys.tafkir.train.diffusion.opd.DefaultDiffusionOpdTrainer.builder();
    }

    public static Acceleration.BackendStatus accelerationStatus() {
        return Acceleration.status();
    }

    public static Acceleration.BackendStatus accelerationStatus(String deviceId) {
        return Acceleration.status(deviceId);
    }
}
