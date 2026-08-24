package tech.kayys.tafkir.ml.timeseries.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Supervised time-series dataset built from a sliding window over a univariate series.
 *
 * <p>Each sample is a pair of {@code (input[lookback], target[horizon])} arrays.
 * The total number of samples is {@code max(0, series.length - lookback - horizon + 1)}.
 *
 * <pre>{@code
 * double[] series = ...;
 * TimeSeriesDataset ds = new TimeSeriesDataset(series, lookback=24, horizon=6);
 * for (int i = 0; i < ds.size(); i++) {
 *     double[] x = ds.getInput(i);   // shape [lookback]
 *     double[] y = ds.getTarget(i);  // shape [horizon]
 * }
 * }</pre>
 */
public final class TimeSeriesDataset {

    private final int lookback;
    private final int horizon;
    private final double[][] inputs;
    private final double[][] targets;

    /**
     * @param series   Univariate time series.
     * @param lookback Number of past steps used as input features.
     * @param horizon  Number of future steps to predict.
     */
    public TimeSeriesDataset(double[] series, int lookback, int horizon) {
        Objects.requireNonNull(series, "series must not be null");
        if (lookback <= 0) throw new IllegalArgumentException("lookback must be > 0");
        if (horizon <= 0)  throw new IllegalArgumentException("horizon must be > 0");

        this.lookback = lookback;
        this.horizon  = horizon;

        int numSamples = Math.max(0, series.length - lookback - horizon + 1);
        this.inputs  = new double[numSamples][];
        this.targets = new double[numSamples][];

        for (int i = 0; i < numSamples; i++) {
            inputs[i]  = Arrays.copyOfRange(series, i, i + lookback);
            targets[i] = Arrays.copyOfRange(series, i + lookback, i + lookback + horizon);
        }
    }

    /** Number of samples in this dataset. */
    public int size() { return inputs.length; }

    /** Input window at index {@code idx} (defensive copy). */
    public double[] getInput(int idx)  { return inputs[idx].clone(); }

    /** Target window at index {@code idx} (defensive copy). */
    public double[] getTarget(int idx) { return targets[idx].clone(); }

    public int lookback() { return lookback; }
    public int horizon()  { return horizon; }

    @Override
    public String toString() {
        return "TimeSeriesDataset[samples=" + size()
                + ", lookback=" + lookback + ", horizon=" + horizon + "]";
    }
}
