///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-aljabr:0.3.0-SNAPSHOT
//DEPS tech.kayys.tafkir:tafkir-trainer-aljabr:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.TafkirTrainer;
import tech.kayys.tafkir.trainer.loss.TafkirMSELoss;
import tech.kayys.tafkir.trainer.model.*;
import tech.kayys.tafkir.trainer.optim.TafkirAdam;

/**
 * XOR problem: the "Hello World" of neural networks.
 */
public class hello_tafkir {
    public static void main(String[] args) {
        TafkirTensor x = TafkirTensor.of(new float[]{
            0, 0, 0, 1, 1, 0, 1, 1
        }, 4, 2);

        TafkirTensor y = TafkirTensor.of(new float[]{
            0, 1, 1, 0
        }, 4, 1);

        TafkirSequential model = new TafkirSequential(
            new TafkirLinear(2, 2),
            new TafkirReLU(),
            new TafkirLinear(2, 1)
        );

        System.out.println("XOR Training with Tafkir + Aljabr");
        System.out.println(model);
        System.out.println();

        TafkirTrainer trainer = new TafkirTrainer(
            model, new TafkirMSELoss(), new TafkirAdam(model.parameters(), 0.1f), 1000, false
        );
        trainer.fit(x, y);

        model.eval();
        TafkirTensor pred = model.forward(x);
        System.out.println("\nFinal predictions:");
        float[] predictions = pred.data();
        for (int i = 0; i < 4; i++) {
            System.out.printf("  XOR(%.0f, %.0f) = %.4f (expected: %.0f)%n",
                x.data()[i * 2], x.data()[i * 2 + 1], predictions[i], y.data()[i]);
        }

        float loss = trainer.evaluate(x, y);
        System.out.printf("\nFinal loss: %.6f%n", loss);
        if (loss < 0.01) {
            System.out.println("SUCCESS: XOR learned!");
        } else {
            System.out.println("FAILURE: XOR not learned.");
            System.exit(1);
        }
    }
}
