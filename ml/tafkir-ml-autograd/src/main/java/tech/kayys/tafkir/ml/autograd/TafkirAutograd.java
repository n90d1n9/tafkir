package tech.kayys.tafkir.ml.autograd;

import tech.kayys.aljabr.autograd.AutogradEngine;
import tech.kayys.aljabr.autograd.GradRegistry;
import tech.kayys.aljabr.ir.GGraph;
import tech.kayys.aljabr.ir.GValueId;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Tafkir-facing autograd wrapper around Aljabr's {@link AutogradEngine}.
 */
public final class TafkirAutograd {

    private static final AutogradEngine ENGINE = new AutogradEngine(GradRegistry.getInstance());

    private TafkirAutograd() {}

    public static void backward(TafkirTensor loss) {
        loss.backward();
    }

    public static void backward(TafkirTensor loss, TafkirTensor seedGrad) {
        loss.setGrad(seedGrad.unwrap());
        loss.backward();
    }

    public static GGraph buildBackward(GGraph forward, GValueId lossId) {
        return ENGINE.buildBackward(forward, lossId);
    }

    public static void zeroGrad(List<TafkirTensor> parameters) {
        for (TafkirTensor p : parameters) {
            if (p.requiresGrad()) {
                p.setGrad(null);
            }
        }
    }
}
