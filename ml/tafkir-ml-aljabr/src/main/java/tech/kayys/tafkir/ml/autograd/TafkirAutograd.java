package tech.kayys.tafkir.ml.autograd;

import tech.kayys.aljabr.autograd.AutogradEngine;
import tech.kayys.aljabr.autograd.GradRegistry;
import tech.kayys.aljabr.ir.GGraph;
import tech.kayys.aljabr.ir.GOp;
import tech.kayys.aljabr.ir.GValueId;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.HashMap;
import java.util.Map;

/**
 * Tafkir-facing autograd wrapper around Aljabr's {@link AutogradEngine}.
 *
 * <p>Aljabr has two autograd modes:
 * <ol>
 *   <li><b>Eager mode</b> (DefaultTensor): Each op stores its own gradFn. Calling
 *       {@code backward()} traverses the chain recursively.</li>
 *   <li><b>Graph mode</b> (LazyTensor): Operations are recorded into a DAG
 *       ({@link GGraph}). The graph is optimized and executed by
 *       {@link tech.kayys.aljabr.core.tensor.lazy.GraphExecutor}.</li>
 * </ol>
 *
 * <p>This class bridges both modes, providing a unified {@code backward()} API.
 */
public final class TafkirAutograd {

    private static final AutogradEngine ENGINE = new AutogradEngine(GradRegistry.getInstance());

    private TafkirAutograd() {}

    /**
     * Triggers backward pass on the given loss tensor.
     *
     * <p>For DefaultTensor (eager mode): delegates to the tensor's own backward chain.
     * <p>For LazyTensor (graph mode): builds a backward graph and executes it.
     */
    public static void backward(TafkirTensor loss) {
        loss.backward();
    }

    /**
     * Triggers backward pass with a custom seed gradient.
     * Useful for higher-order derivatives or gradient checkpointing.
     */
    public static void backward(TafkirTensor loss, TafkirTensor seedGrad) {
        // Set the gradient on the loss tensor before backward
        loss.setGrad(seedGrad.unwrap());
        loss.backward();
    }

    /**
     * Low-level API: builds a backward graph from a forward GGraph.
     * This is used internally by the trainer for custom optimization passes.
     */
    public static GGraph buildBackward(GGraph forward, GValueId lossId) {
        return ENGINE.buildBackward(forward, lossId);
    }

    /**
     * Zeroes all gradients in the given parameters.
     */
    public static void zeroGrad(java.util.List<TafkirTensor> parameters) {
        for (TafkirTensor p : parameters) {
            if (p.requiresGrad()) {
                p.setGrad(null);
            }
        }
    }
}
