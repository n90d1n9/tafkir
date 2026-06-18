package tech.kayys.tafkir.ml;

import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.NNModule;
import tech.kayys.tafkir.ml.nn.layer.Sequential;

/**
 * Tensor and module construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlTensorFactoryFacade extends AljabrDlDataFacade {
    protected AljabrDlTensorFactoryFacade() {
    }

    public static GradTensor tensor(float[] data, long... shape) {
        return Aljabr.tensor(data, shape);
    }

    public static NNModule sequential(NNModule... layers) {
        return new Sequential(layers);
    }
}
