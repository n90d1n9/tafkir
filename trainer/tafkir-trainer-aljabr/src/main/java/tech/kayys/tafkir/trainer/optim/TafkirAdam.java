package tech.kayys.tafkir.trainer.optim;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adam optimizer with in-place operations.
 */
public final class TafkirAdam implements TafkirOptimizer {

    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final Map<TafkirTensor, TafkirTensor> m = new HashMap<>();
    private final Map<TafkirTensor, TafkirTensor> v = new HashMap<>();
    private int t = 0;

    public TafkirAdam(List<TafkirTensor> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, 0.0f);
    }

    public TafkirAdam(List<TafkirTensor> parameters, float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        for (TafkirTensor p : parameters) {
            m.put(p, TafkirTensor.zeros(p.shapeArray()));
            v.put(p, TafkirTensor.zeros(p.shapeArray()));
        }
    }

    @Override
    public void step(List<TafkirTensor> parameters) {
        t++;
        float biasCorr1 = 1 - (float) Math.pow(beta1, t);
        float biasCorr2 = 1 - (float) Math.pow(beta2, t);

        for (TafkirTensor p : parameters) {
            if (!p.requiresGrad()) continue;
            TafkirTensor grad = p.gradTensor();
            if (grad == null) continue;

            if (weightDecay != 0) {
                grad.add_(p.mul(weightDecay));
            }

            TafkirTensor mt = m.get(p);
            TafkirTensor vt = v.get(p);

            mt.mul_(beta1);
            mt.add_(grad.mul(1 - beta1));

            vt.mul_(beta2);
            vt.add_(grad.mul(grad).mul(1 - beta2));

            TafkirTensor mHat = mt; // alias
            mHat.div_(biasCorr1);

            TafkirTensor vHat = vt; // alias
            vHat.div_(biasCorr2);
            vHat.sqrt_();
            vHat.add_(eps);

            mHat.div_(vHat);
            p.sub_(mHat.mul(lr));

            // Restore moment estimates
            mt.mul_(biasCorr1);
            vt.mul_(biasCorr2);
            vt.sub_(eps);
            vt.mul_(vHat);
        }
    }

    @Override
    public void zeroGrad(List<TafkirTensor> parameters) {
        for (TafkirTensor p : parameters) {
            if (p.requiresGrad()) p.setGrad(null);
        }
    }
}
