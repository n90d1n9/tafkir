package tech.kayys.tafkir.trainer;

import tech.kayys.tafkir.ml.autograd.TafkirAutograd;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.loss.TafkirLoss;
import tech.kayys.tafkir.trainer.model.TafkirModel;
import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

import java.util.List;

/**
 * Real training loop using Aljabr backends.
 *
 * <p>Example usage:
 * <pre>{@code
 * TafkirSequential model = new TafkirSequential(
 *     new TafkirLinear(784, 256),
 *     new TafkirReLU(),
 *     new TafkirLinear(256, 10)
 * );
 *
 * TafkirTrainer trainer = new TafkirTrainer(
 *     model,
 *     new TafkirCrossEntropyLoss(),
 *     new TafkirAdam(model.parameters(), 0.001f),
 *     10
 * );
 *
 * trainer.fit(trainImages, trainLabels);
 * }</pre>
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

    /**
     * Trains the model on the given data for the configured number of epochs.
     *
     * @param x input features
     * @param y target labels
     */
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
            // Forward pass
            TafkirTensor pred = model.forward(x);
            TafkirTensor loss = lossFn.compute(pred, y);

            // Backward pass
            TafkirAutograd.backward(loss);

            // Optimizer step
            optimizer.step(params);

            // Zero gradients for next iteration
            optimizer.zeroGrad(params);

            if (verbose) {
                System.out.printf("Epoch %d/%d, loss: %.6f%n", epoch + 1, epochs, loss.item());
            }
        }

        if (verbose) {
            System.out.println("\nTraining complete.");
        }
    }

    /**
     * Evaluates the model on validation data.
     */
    public float evaluate(TafkirTensor x, TafkirTensor y) {
        model.eval();
        TafkirTensor pred = model.forward(x);
        TafkirTensor loss = lossFn.compute(pred, y);
        return loss.item();
    }
}
