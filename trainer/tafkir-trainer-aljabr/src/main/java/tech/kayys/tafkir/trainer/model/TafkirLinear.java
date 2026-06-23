package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Fully connected layer: y = x @ W^T + b
 *
 * <p>Weights are initialized with Kaiming uniform (He initialization).
 */
public final class TafkirLinear implements TafkirLayer {

    private final TafkirTensor weight;
    private final TafkirTensor bias;
    private final int inFeatures;
    private final int outFeatures;

    public TafkirLinear(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        // Kaiming uniform initialization
        float bound = (float) Math.sqrt(1.0 / inFeatures);
        this.weight = TafkirTensor.rand(outFeatures, inFeatures)
            .mul(2 * bound)
            .sub(bound)
            .requiresGrad(true);

        this.bias = TafkirTensor.zeros(outFeatures).requiresGrad(true);
    }

    @Override
    public TafkirTensor forward(TafkirTensor input, boolean training) {
        // input: [..., inFeatures]
        // weight: [outFeatures, inFeatures]
        // output: input @ weight^T + bias
        TafkirTensor output = input.matmul(weight.transpose());
        // Add bias with broadcasting
        return output.add(bias);
    }

    @Override
    public List<TafkirTensor> parameters() {
        return List.of(weight, bias);
    }

    public int inFeatures() { return inFeatures; }
    public int outFeatures() { return outFeatures; }

    @Override
    public String toString() {
        return "Linear(" + inFeatures + ", " + outFeatures + ")";
    }
}
