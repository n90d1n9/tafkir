package tech.kayys.tafkir.trainer;

import tech.kayys.tafkir.ml.autograd.TafkirAutograd;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.loss.TafkirLoss;
import tech.kayys.tafkir.trainer.model.TafkirModel;
import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

import java.util.List;

/**
 * Real training loop using Aljabr backends.
 */
public final class TafkirTrainer {

    private final TafkirModel model;
    private final TafkirLoss lossFn;
    private final TafkirOptimizer optimizer;
    private final int epochs;
    private final boolean verbose;

    public TafkirTrainer(TafkirModel model, TafkirLoss lossFn, TafkirOptimizer optimizer, int epochs) {
        this(model, lossFn, optimizer, epochs, true);
    }

    public TafkirTrainer(TafkirModel model, TafkirLoss lossFn, TafkirOptimizer optimizer, int epochs, boolean verbose) {
        this.model = model;
        this.lossFn = lossFn;
        this.optimizer = optimizer;
        this.epochs = epochs;
        this.verbose = verbose;
    }

    public void fit(TafkirTensor x, TafkirTensor y) {
        model.train();
        List<TafkirTensor> params = model.parameters();

        if (verbose) {
            System.out.println("Starting training...");
            System.out.println("Model: " + model.getClass().getSimpleName());
            System.out.println("Parameters: " + params.size());
            System.out.println("Epochs: " + epochs);
            System.out.println();
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            TafkirTensor pred = model.forward(x);
            TafkirTensor loss = lossFn.compute(pred, y);

            TafkirAutograd.backward(loss);
            optimizer.step(params);
            optimizer.zeroGrad(params);

            if (verbose) {
                System.out.printf("Epoch %d/%d, loss: %.6f%n", epoch + 1, epochs, loss.item());
            }
        }

        if (verbose) System.out.println("\nTraining complete.");
    }

    public float evaluate(TafkirTensor x, TafkirTensor y) {
        model.eval();
        TafkirTensor pred = model.forward(x);
        TafkirTensor loss = lossFn.compute(pred, y);
        return loss.item();
    }
}
