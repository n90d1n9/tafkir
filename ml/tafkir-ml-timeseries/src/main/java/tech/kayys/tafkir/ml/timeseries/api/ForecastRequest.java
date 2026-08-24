package tech.kayys.tafkir.ml.timeseries.api;

import java.util.Objects;

/**
 * Immutable request for a one-shot forecast.
 *
 * @param series  Univariate time series (must be non-empty).
 * @param horizon Number of future steps to forecast (must be &gt; 0).
 */
public record ForecastRequest(double[] series, int horizon) {

    public ForecastRequest {
        Objects.requireNonNull(series, "series must not be null");
        if (series.length == 0) throw new IllegalArgumentException("series must not be empty");
        if (horizon <= 0)       throw new IllegalArgumentException("horizon must be > 0, got: " + horizon);
    }

    /** Convenience factory. */
    public static ForecastRequest of(double[] series, int horizon) {
        return new ForecastRequest(series.clone(), horizon);
    }

    /** Returns a defensive copy of the series array. */
    @Override
    public double[] series() {
        return series.clone();
    }
}
