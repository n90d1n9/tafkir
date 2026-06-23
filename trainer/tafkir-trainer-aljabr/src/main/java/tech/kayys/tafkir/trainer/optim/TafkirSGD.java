package tech.kayys.tafkir.trainer.optim;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Stochastic Gradient Descent with optional momentum.
 * Uses in-place operations to avoid allocating new tensors.
 */
public final class TafkirSGD implements TafkirOptimizer {

    private final float lr;
    private final float momentum;
    private final float weightDecay;
    private final boolean nesterov;
    private final Map<TafkirTensor, TafkirTensor> velocities = new HashMap<>();

    public TafkirSGD(List<TafkirTensor> parameters, float lr) {
        this(parameters, lr, 0.0f, 0.0f, false);
    }

    public TafkirSGD(List<TafkirTensor> parameters, float lr, float momentum, float weightDecay, boolean nesterov) {
        this.lr = lr;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
        for (TafkirTensor p : parameters) {
            velocities.put(p, TafkirTensor.zeros(p.shapeArray()));
        }
    }

    @Override
    public void step(List<TafkirTensor> parameters) {
        for (TafkirTensor p : parameters) {
            if (!p.requiresGrad()) continue;
            TafkirTensor grad = p.gradTensor();
            if (grad == null) continue;

            if (weightDecay != 0) {
                grad.add_(p.mul(weightDecay));
            }

            TafkirTensor v = velocities.get(p);
            if (momentum > 0) {
                v.mul_(momentum);
                v.add_(grad);
                if (nesterov) {
                    TafkirTensor update = v.mul(momentum).add(grad);
                    p.sub_(update.mul(lr));
                } else {
                    p.sub_(v.mul(lr));
                }
            } else {
                p.sub_(grad.mul(lr));
            }
        }
    }

    @Override
    public void zeroGrad(List<TafkirTensor> parameters) {
        for (TafkirTensor p : parameters) {
            if (p.requiresGrad()) p.setGrad(null);
        }
    }
}
