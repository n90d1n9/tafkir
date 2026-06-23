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
 * MNIST MLP training example.
 */
public class mnist_mlp {
    public static void main(String[] args) throws IOException {
        System.out.println("=== Tafkir MNIST MLP Training ===\n");

        MNISTDataset trainSet = MNISTDataset.loadTrain("data/mnist");
        MNISTDataset testSet = MNISTDataset.loadTest("data/mnist");
        System.out.println("Train: " + trainSet.numSamples() + ", Test: " + testSet.numSamples());

        TafkirSequential model = new TafkirSequential(
            new TafkirLinear(784, 256),
            new TafkirReLU(),
            new TafkirLinear(256, 128),
            new TafkirReLU(),
            new TafkirLinear(128, 10)
        );
        System.out.println("\n" + model);

        TafkirTrainer trainer = new TafkirTrainer(
            model,
            new TafkirCrossEntropyLoss(),
            new TafkirAdam(model.parameters(), 0.001f),
            10
        );

        // Train on full dataset (no batching for simplicity in example)
        trainer.fit(trainSet.images(), trainSet.labels());

        float testLoss = trainer.evaluate(testSet.images(), testSet.labels());
        System.out.printf("\nTest loss: %.4f%n", testLoss);
    }
}
