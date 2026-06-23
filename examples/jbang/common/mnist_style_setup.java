///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-aljabr:0.3.0-SNAPSHOT
//DEPS tech.kayys.tafkir:tafkir-trainer-aljabr:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.TafkirTrainer;
import tech.kayys.tafkir.trainer.loss.TafkirCrossEntropyLoss;
import tech.kayys.tafkir.trainer.model.*;
import tech.kayys.tafkir.trainer.optim.TafkirAdam;

public class mnist_style_setup {
    public static void main(String[] args) {
        int inputSize = 28 * 28;   // MNIST image flattened
        int hiddenSize = 256;
        int numClasses = 10;
        int batchSize = 64;

        // 1. Build model
        TafkirSequential model = new TafkirSequential(
            new TafkirLinear(inputSize, hiddenSize),
            new TafkirReLU(),
            new TafkirLinear(hiddenSize, hiddenSize / 2),
            new TafkirReLU(),
            new TafkirLinear(hiddenSize / 2, numClasses)
        );

        System.out.println(model);
        System.out.println();

        // 2. Training components
        TafkirCrossEntropyLoss criterion = new TafkirCrossEntropyLoss();
        TafkirAdam optimizer = new TafkirAdam(model.parameters(), 0.001f);

        // 3. Fake MNIST-like data (replace with real data loading)
        TafkirTensor fakeImages = TafkirTensor.randn(batchSize, inputSize);
        TafkirTensor fakeLabels = TafkirTensor.zeros(batchSize); // class indices 0-9

        // 4. Train
        TafkirTrainer trainer = new TafkirTrainer(model, criterion, optimizer, 5);
        trainer.fit(fakeImages, fakeLabels);

        // 5. Evaluate
        model.eval();
        TafkirTensor valOutput = model.forward(fakeImages);
        System.out.println("\nEval output shape: " + java.util.Arrays.toString(valOutput.shapeArray()));
        System.out.println("Sample predictions (first 5):");
        for (int i = 0; i < 5; i++) {
            System.out.println("  Image " + i + ": class probabilities shape = " +
                java.util.Arrays.toString(valOutput.shapeArray()));
        }
    }
}
