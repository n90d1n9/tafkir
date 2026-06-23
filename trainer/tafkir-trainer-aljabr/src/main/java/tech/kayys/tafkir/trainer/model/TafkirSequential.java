package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Sequential model composed of layers.
 */
public final class TafkirSequential implements TafkirModel {

    private final List<TafkirLayer> layers;
    private boolean training = true;

    public TafkirSequential(TafkirLayer... layers) {
        this.layers = List.of(layers);
    }

    @Override
    public TafkirTensor forward(TafkirTensor input) {
        TafkirTensor x = input;
        for (TafkirLayer layer : layers) {
            x = layer.forward(x, training);
        }
        return x;
    }

    @Override
    public List<TafkirTensor> parameters() {
        List<TafkirTensor> params = new ArrayList<>();
        for (TafkirLayer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public void train() { this.training = true; }

    @Override
    public void eval() { this.training = false; }

    @Override
    public boolean isTraining() { return training; }

    public long parameterCount() {
        return parameters().stream().mapToLong(TafkirTensor::numel).sum();
    }

    public String parameterCountFormatted() {
        long count = parameterCount();
        if (count < 1_000) return String.valueOf(count);
        if (count < 1_000_000) return String.format("%.1fK", count / 1_000.0);
        if (count < 1_000_000_000) return String.format("%.1fM", count / 1_000_000.0);
        return String.format("%.1fB", count / 1_000_000_000.0);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("TafkirSequential:\n");
        for (int i = 0; i < layers.size(); i++) {
            sb.append("  [").append(i).append("] ").append(layers.get(i)).append("\n");
        }
        sb.append("Total params: ").append(parameterCountFormatted());
        return sb.toString();
    }
}
