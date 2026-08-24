package tech.kayys.tafkir.ml.timeseries.statistical;

import tech.kayys.tafkir.ml.timeseries.api.Forecaster;

/**
 * Exponential Smoothing family: Simple (SES), Holt's Linear Trend, and Holt-Winters Additive.
 *
 * <p>Select the model via {@link Mode}.
 *
 * <pre>{@code
 * // Simple (level only)
 * Forecaster ses = new ExponentialSmoothing(Mode.SIMPLE, 0.3, 0, 0, 0);
 *
 * // Holt – level + trend
 * Forecaster holt = new ExponentialSmoothing(Mode.HOLT, 0.4, 0.1, 0, 0);
 *
 * // Holt-Winters additive – level + trend + seasonality
 * Forecaster hw = new ExponentialSmoothing(Mode.HOLT_WINTERS, 0.3, 0.1, 0.2, 12);
 * }</pre>
 */
public final class ExponentialSmoothing implements Forecaster {

    /** Selects the exponential smoothing variant. */
    public enum Mode { SIMPLE, HOLT, HOLT_WINTERS }

    private final Mode   mode;
    private final double alpha;   // level smoothing (0 < α ≤ 1)
    private final double beta;    // trend smoothing (0 < β ≤ 1), used in HOLT / HOLT_WINTERS
    private final double gamma;   // seasonal smoothing (0 < γ ≤ 1), used in HOLT_WINTERS
    private final int    period;  // seasonal period, used in HOLT_WINTERS

    // Fitted state
    private double   level;
    private double   trend;
    private double[] seasonals;  // length == period for HOLT_WINTERS
    private int      seriesLen;

    /**
     * @param mode   Smoothing variant.
     * @param alpha  Level smoothing coefficient.
     * @param beta   Trend smoothing coefficient (ignored for SIMPLE).
     * @param gamma  Seasonal smoothing coefficient (ignored unless HOLT_WINTERS).
     * @param period Seasonal period (ignored unless HOLT_WINTERS; must be >= 2).
     */
    public ExponentialSmoothing(Mode mode, double alpha, double beta, double gamma, int period) {
        this.mode   = mode;
        this.alpha  = checkParam(alpha, "alpha");
        this.beta   = checkParam(beta, "beta");
        this.gamma  = checkParam(gamma, "gamma");
        if (mode == Mode.HOLT_WINTERS && period < 2)
            throw new IllegalArgumentException("period must be >= 2 for Holt-Winters, got: " + period);
        this.period = period;
    }

    /** Convenience: Simple Exponential Smoothing with the given alpha. */
    public static ExponentialSmoothing simple(double alpha) {
        return new ExponentialSmoothing(Mode.SIMPLE, alpha, 0, 0, 0);
    }

    /** Convenience: Holt's linear trend model. */
    public static ExponentialSmoothing holt(double alpha, double beta) {
        return new ExponentialSmoothing(Mode.HOLT, alpha, beta, 0, 0);
    }

    /** Convenience: Holt-Winters additive seasonal model. */
    public static ExponentialSmoothing holtWinters(double alpha, double beta, double gamma, int period) {
        return new ExponentialSmoothing(Mode.HOLT_WINTERS, alpha, beta, gamma, period);
    }

    // ── Fit ──────────────────────────────────────────────────────────────────

    @Override
    public void fit(double[] series) {
        if (series == null || series.length < 2)
            throw new IllegalArgumentException("Series must have >= 2 observations");
        if (mode == Mode.HOLT_WINTERS && series.length < 2 * period)
            throw new IllegalArgumentException(
                    "Holt-Winters needs >= 2 seasonal periods (" + (2 * period) + " obs), got " + series.length);

        seriesLen = series.length;
        switch (mode) {
            case SIMPLE       -> fitSimple(series);
            case HOLT         -> fitHolt(series);
            case HOLT_WINTERS -> fitHoltWinters(series);
        }
    }

    private void fitSimple(double[] y) {
        level = y[0];
        for (int t = 1; t < y.length; t++) {
            level = alpha * y[t] + (1 - alpha) * level;
        }
        trend = 0; seasonals = null;
    }

    private void fitHolt(double[] y) {
        level = y[0];
        trend = y[1] - y[0];
        for (int t = 1; t < y.length; t++) {
            double prevLevel = level;
            level = alpha * y[t] + (1 - alpha) * (level + trend);
            trend = beta * (level - prevLevel) + (1 - beta) * trend;
        }
        seasonals = null;
    }

    private void fitHoltWinters(double[] y) {
        // Initialise level and trend from first two seasonal averages
        double s1 = 0, s2 = 0;
        for (int i = 0; i < period; i++) s1 += y[i];
        for (int i = period; i < 2 * period; i++) s2 += y[i];
        s1 /= period; s2 /= period;
        level = s1;
        trend = (s2 - s1) / period;

        // Initialise seasonal components as deviation from first-period mean
        seasonals = new double[period];
        double avg = 0;
        for (int i = 0; i < period; i++) avg += y[i];
        avg /= period;
        for (int i = 0; i < period; i++) seasonals[i] = y[i] - avg;

        // Update pass
        for (int t = 0; t < y.length; t++) {
            int sIdx = t % period;
            double prevLevel = level;
            level    = alpha  * (y[t] - seasonals[sIdx]) + (1 - alpha) * (level + trend);
            trend    = beta   * (level - prevLevel)       + (1 - beta)  * trend;
            seasonals[sIdx] = gamma * (y[t] - level) + (1 - gamma) * seasonals[sIdx];
        }
    }

    // ── Predict ──────────────────────────────────────────────────────────────

    @Override
    public double[] predict(int horizon) {
        if (seriesLen == 0) throw new IllegalStateException("Call fit() before predict()");
        double[] result = new double[horizon];
        switch (mode) {
            case SIMPLE -> {
                for (int h = 1; h <= horizon; h++) result[h - 1] = level;
            }
            case HOLT -> {
                for (int h = 1; h <= horizon; h++) result[h - 1] = level + h * trend;
            }
            case HOLT_WINTERS -> {
                for (int h = 1; h <= horizon; h++) {
                    int sIdx = (seriesLen + h - 1) % period;
                    result[h - 1] = level + h * trend + seasonals[sIdx];
                }
            }
        }
        return result;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    @Override public String name() { return "ExponentialSmoothing(" + mode + ")"; }
    public Mode   mode()      { return mode; }
    public double alpha()     { return alpha; }
    public double beta()      { return beta; }
    public double gamma()     { return gamma; }
    public int    period()    { return period; }
    public double getLevel()  { return level; }
    public double getTrend()  { return trend; }
    public double[] getSeasonals() { return (seasonals == null) ? null : seasonals.clone(); }

    private static double checkParam(double v, String name) {
        if (v < 0.0 || v > 1.0)
            throw new IllegalArgumentException(name + " must be in [0, 1], got: " + v);
        return v;
    }
}
