package tech.kayys.tafkir.ml.timeseries.api;

/**
 * Unified forecaster SPI for all time-series models (statistical + neural).
 *
 * <p>Minimal contract:
 * <ol>
 *   <li>{@link #fit(double[])} — train/calibrate on a univariate series.</li>
 *   <li>{@link #predict(int)} — produce {@code horizon} point forecasts.</li>
 * </ol>
 *
 * <p>Use the default {@link #forecast(ForecastRequest)} convenience method to
 * combine fit + predict in one call.
 */
public interface Forecaster {

    /** Train or calibrate the model on {@code series}. */
    void fit(double[] series);

    /**
     * Return {@code horizon} point forecasts after the last observed value.
     * Must be called after {@link #fit(double[])}.
     */
    double[] predict(int horizon);

    /** Human-readable model name used in {@link ForecastResult}. */
    default String name() {
        return getClass().getSimpleName();
    }

    /**
     * Convenience: fit then predict in one call.
     * Implementations may override for performance (e.g., avoid re-fit).
     */
    default ForecastResult forecast(ForecastRequest request) {
        fit(request.series());
        double[] values = predict(request.horizon());
        return new ForecastResult(values, name());
    }
}
