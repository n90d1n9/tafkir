///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-aljabr:0.3.0-SNAPSHOT
//DEPS tech.kayys.tafkir:tafkir-trainer-aljabr:0.3.0-SNAPSHOT
//DEPS tech.kayys.tafkir:tafkir-data:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.tafkir.data.MNISTDataset;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.TafkirTrainer;
import tech.kayys.tafkir.trainer.loss.TafkirCrossEntropyLoss;
import tech.kayys.tafkir.trainer.model.*;
import tech.kayys.tafkir.trainer.optim.TafkirAdam;

import java.io.IOException;

/**
 * MNIST CNN training example.
 */
public class mnist_cnn {
    public static void main(String[] args) throws IOException {
        System.out.println("=== Tafkir MNIST CNN Training ===\n");

        MNISTDataset trainSet = MNISTDataset.loadTrain("data/mnist");
        System.out.println("Train samples: " + trainSet.numSamples());

        // CNN: [B, 1, 28, 28] -> Conv -> Pool -> Conv -> Pool -> Flatten -> FC
        TafkirSequential model = new TafkirSequential(
            // Input needs to be reshaped from [B, 784] to [B, 1, 28, 28]
            // For now, we use a wrapper or reshape in the training loop
            new TafkirConv2d(1, 32, 3, 1, 1, 1, 1),
            new TafkirReLU(),
            new TafkirMaxPool2d(2, 2, 0),
            new TafkirConv2d(32, 64, 3, 1, 1, 1, 1),
            new TafkirReLU(),
            new TafkirMaxPool2d(2, 2, 0),
            new TafkirFlatten(),
            new TafkirLinear(3136, 128),
            new TafkirReLU(),
            new TafkirLinear(128, 10)
        );

        System.out.println("\n" + model);

        TafkirCrossEntropyLoss criterion = new TafkirCrossEntropyLoss();
        TafkirAdam optimizer = new TafkirAdam(model.parameters(), 0.001f);

        // Training with batching
        int epochs = 5;
        int batchSize = 64;
        int numBatches = trainSet.numSamples() / batchSize;

        System.out.println("\nStarting training...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0;
            for (int batch = 0; batch < numBatches; batch++) {
                int start = batch * batchSize;
                int end = start + batchSize;
                TafkirTensor[] b = trainSet.batch(start, end);
                TafkirTensor images = b[0].reshape(batchSize, 1, 28, 28);
                TafkirTensor labels = b[1];

                TafkirTensor pred = model.forward(images);
                TafkirTensor loss = criterion.compute(pred, labels);
                loss.backward();
                optimizer.step(model.parameters());
                optimizer.zeroGrad(model.parameters());
                epochLoss += loss.item();
            }
            System.out.printf("Epoch %d/%d, avg loss: %.4f%n", epoch + 1, epochs, epochLoss / numBatches);
        }
        System.out.println("Training complete!");
    }
}
