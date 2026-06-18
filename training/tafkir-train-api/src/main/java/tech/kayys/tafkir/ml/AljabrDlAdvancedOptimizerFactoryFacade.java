package tech.kayys.tafkir.ml;

import java.util.List;
import tech.kayys.tafkir.ml.nn.Parameter;
import tech.kayys.tafkir.ml.optim.Adadelta;
import tech.kayys.tafkir.ml.optim.Adagrad;
import tech.kayys.tafkir.ml.optim.LAMB;
import tech.kayys.tafkir.ml.optim.Lion;
import tech.kayys.tafkir.ml.optim.Lookahead;
import tech.kayys.tafkir.ml.optim.Optimizer;
import tech.kayys.tafkir.ml.optim.RAdam;
import tech.kayys.tafkir.ml.optim.SAM;

/**
 * Advanced optimizer construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlAdvancedOptimizerFactoryFacade extends AljabrDlAdaptiveOptimizerFactoryFacade {
    protected AljabrDlAdvancedOptimizerFactoryFacade() {
    }

    public static Optimizer lookahead(Optimizer inner) {
        return new Lookahead(inner);
    }

    public static Optimizer lookahead(Optimizer inner, int k, float alpha) {
        return new Lookahead(inner, k, alpha);
    }

    public static Optimizer sam(List<Parameter> params, Optimizer base) {
        return new SAM(params, base);
    }

    public static Optimizer sam(List<Parameter> params, Optimizer base, float rho) {
        return new SAM(params, base, rho);
    }

    public static Optimizer adagrad(List<Parameter> params, float lr) {
        return new Adagrad(params, lr);
    }

    public static Optimizer adagrad(List<Parameter> params, float lr, float eps, float weightDecay) {
        return new Adagrad(params, lr, eps, weightDecay);
    }

    public static Optimizer adadelta(List<Parameter> params) {
        return new Adadelta(params);
    }

    public static Optimizer adadelta(List<Parameter> params, float lr, float rho, float eps) {
        return new Adadelta(params, lr, rho, eps);
    }

    public static Optimizer lamb(List<Parameter> params, float lr) {
        return new LAMB(params, lr);
    }

    public static Optimizer lamb(
            List<Parameter> params,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay) {
        return new LAMB(params, lr, beta1, beta2, eps, weightDecay);
    }

    public static Optimizer lion(List<Parameter> params, float lr) {
        return new Lion(params, lr);
    }

    public static Optimizer lion(
            List<Parameter> params,
            float lr,
            float beta1,
            float beta2,
            float weightDecay) {
        return new Lion(params, lr, beta1, beta2, weightDecay);
    }

    public static Optimizer radam(List<Parameter> params, float lr) {
        return RAdam.builder(params, lr).build();
    }

    public static Optimizer radam(List<Parameter> params, float lr, float weightDecay) {
        return RAdam.builder(params, lr).weightDecay(weightDecay).build();
    }
}
