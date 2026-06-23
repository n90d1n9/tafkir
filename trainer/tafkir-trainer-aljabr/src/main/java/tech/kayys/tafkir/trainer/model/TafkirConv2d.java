package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * 2D Convolutional layer.
 */
public final class TafkirConv2d implements TafkirLayer {

    private final TafkirTensor weight;
    private final TafkirTensor bias;
    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;

    public TafkirConv2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, 1, 1);
    }

    public TafkirConv2d(int inChannels, int outChannels, int kernelSize,
                         int stride, int padding, int dilation, int groups) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;

        if (inChannels % groups != 0) {
            throw new IllegalArgumentException("inChannels must be divisible by groups");
        }
        if (outChannels % groups != 0) {
            throw new IllegalArgumentException("outChannels must be divisible by groups");
        }

        float fanIn = (float) (inChannels / groups) * kernelSize * kernelSize;
        float bound = (float) Math.sqrt(1.0 / fanIn);

        this.weight = TafkirTensor.rand(outChannels, inChannels / groups, kernelSize, kernelSize)
            .mul(2 * bound)
            .sub(bound)
            .requiresGrad(true);

        this.bias = TafkirTensor.zeros(outChannels).requiresGrad(true);
    }

    @Override
    public TafkirTensor forward(TafkirTensor input, boolean training) {
        return input.conv2d(weight, bias, stride, padding, dilation, groups);
    }

    @Override
    public List<TafkirTensor> parameters() {
        return List.of(weight, bias);
    }

    public int inChannels() { return inChannels; }
    public int outChannels() { return outChannels; }
    public int kernelSize() { return kernelSize; }
    public int stride() { return stride; }
    public int padding() { return padding; }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel=%d, stride=%d, padding=%d)",
            inChannels, outChannels, kernelSize, stride, padding);
    }
}
