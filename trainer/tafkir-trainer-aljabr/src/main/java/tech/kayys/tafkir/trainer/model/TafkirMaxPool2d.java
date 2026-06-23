package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

public final class TafkirMaxPool2d implements TafkirLayer {

    private final int kernelSize;
    private final int stride;
    private final int padding;

    public TafkirMaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    public TafkirMaxPool2d(int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public TafkirTensor forward(TafkirTensor input, boolean training) {
        return input.maxPool2d(kernelSize, stride, padding);
    }

    @Override
    public List<TafkirTensor> parameters() {
        return List.of();
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel=%d, stride=%d, padding=%d)", kernelSize, stride, padding);
    }
}
