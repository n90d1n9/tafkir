package tech.kayys.tafkir.ml.timeseries.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable result from a {@link Forecaster}.
 *
 * @param values    Point forecast values (length == horizon).
 * @param modelName Name of the model that produced this result.
 */
public record ForecastResult(double[] values, String modelName) {

    public ForecastResult {
        Objects.requireNonNull(values, "values must not be null");
        Objects.requireNonNull(modelName, "modelName must not be null");
        if (values.length == 0) throw new IllegalArgumentException("values must not be empty");
    }

    /** Number of forecast steps (i.e. the horizon). */
    public int horizon() { return values.length; }

    /** Point forecast at step {@code step} (0-based). */
    public double valueAt(int step) {
        if (step < 0 || step >= values.length)
            throw new IndexOutOfBoundsException("step=" + step + " horizon=" + values.length);
        return values[step];
    }

    /** Returns a defensive copy of the values array. */
    @Override
    public double[] values() { return values.clone(); }

    @Override
    public String toString() {
        return "ForecastResult[model=" + modelName
                + ", horizon=" + values.length
                + ", values=" + Arrays.toString(values) + "]";
    }
}
