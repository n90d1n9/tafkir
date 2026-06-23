package tech.kayys.tafkir.trainer;

import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.loss.TafkirMSELoss;
import tech.kayys.tafkir.trainer.model.TafkirLinear;
import tech.kayys.tafkir.trainer.model.TafkirReLU;
import tech.kayys.tafkir.trainer.model.TafkirSequential;
import tech.kayys.tafkir.trainer.optim.TafkirAdam;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test: verifies that Tafkir can actually train a neural network.
 */
public class XORTrainingTest {

    @Test
    public void testXorConvergence() {
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

        TafkirTrainer trainer = new TafkirTrainer(
            model, new TafkirMSELoss(), new TafkirAdam(model.parameters(), 0.1f), 1000, false
        );

        trainer.fit(x, y);

        float finalLoss = trainer.evaluate(x, y);
        assertTrue(finalLoss < 0.01, "XOR should converge to loss < 0.01, got " + finalLoss);

        model.eval();
        TafkirTensor pred = model.forward(x);
        float[] p = pred.data();
        float[] expected = y.data();
        for (int i = 0; i < 4; i++) {
            assertEquals(expected[i], p[i], 0.1, "Prediction " + i + " should be close to " + expected[i]);
        }
    }

    @Test
    public void testParameterCount() {
        TafkirSequential model = new TafkirSequential(
            new TafkirLinear(784, 256),
            new TafkirReLU(),
            new TafkirLinear(256, 10)
        );
        assertEquals(4, model.parameters().size());
        assertEquals(784L * 256 + 256 + 256L * 10 + 10, model.parameterCount());
    }

    @Test
    public void testTensorCreation() {
        TafkirTensor t = TafkirTensor.zeros(2, 3, 4);
        assertArrayEquals(new long[]{2, 3, 4}, t.shapeArray());
        assertEquals(24, t.numel());

        TafkirTensor r = TafkirTensor.randn(5, 5);
        assertEquals(25, r.numel());
        assertTrue(r.requiresGrad() == false);
    }

    @Test
    public void testTensorOperations() {
        TafkirTensor a = TafkirTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        TafkirTensor b = TafkirTensor.of(new float[]{1, 1, 1, 1}, 2, 2);
        TafkirTensor c = a.add(b);
        assertArrayEquals(new float[]{2, 3, 4, 5}, c.data(), 0.001f);
    }

    @Test
    public void testMatmul() {
        TafkirTensor a = TafkirTensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        TafkirTensor b = TafkirTensor.of(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        TafkirTensor c = a.matmul(b);
        assertArrayEquals(new long[]{2, 2}, c.shapeArray());
    }
}
