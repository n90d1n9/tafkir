package tech.kayys.tafkir.trainer.model;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

public final class TafkirFlatten implements TafkirLayer {
    @Override
    public TafkirTensor forward(TafkirTensor input, boolean training) {
        return input.flatten();
    }

    @Override
    public List<TafkirTensor> parameters() {
        return List.of();
    }

    @Override
    public String toString() {
        return "Flatten()";
    }
}
