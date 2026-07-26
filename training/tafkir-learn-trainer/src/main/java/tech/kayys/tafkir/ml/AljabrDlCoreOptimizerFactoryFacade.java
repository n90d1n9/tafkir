package tech.kayys.tafkir.ml;

import java.util.List;
import tech.kayys.tafkir.ml.nn.Parameter;
import tech.kayys.tafkir.ml.optim.Optimizer;
import tech.kayys.tafkir.ml.optim.SGD;

/**
 * Core optimizer construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlCoreOptimizerFactoryFacade extends AljabrDlTensorFactoryFacade {
    protected AljabrDlCoreOptimizerFactoryFacade() {
    }

    public static Optimizer sgd(List<Parameter> params, float lr) {
        return SGD.builder(params, lr).build();
    }

    public static Optimizer sgd(List<Parameter> params, float lr, float momentum) {
        return SGD.builder(params, lr)
                .momentum(momentum)
                .build();
    }

    public static Optimizer sgd(
            List<Parameter> params,
            float lr,
            float momentum,
            float weightDecay,
            boolean nesterov) {
        return SGD.builder(params, lr)
                .momentum(momentum)
                .weightDecay(weightDecay)
                .nesterov(nesterov)
                .build();
    }
}
