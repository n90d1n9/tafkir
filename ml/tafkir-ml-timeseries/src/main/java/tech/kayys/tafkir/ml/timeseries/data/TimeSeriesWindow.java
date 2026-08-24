package tech.kayys.tafkir.ml.timeseries.data;

import java.util.Arrays;

/**
 * Utility for sliding-window operations and lag-feature generation.
 *
 * <p>All methods are static and allocate fresh arrays — no internal state.
 */
public final class TimeSeriesWindow {

    private TimeSeriesWindow() {}

    /**
     * Generate an overlapping sliding window view of the series.
     *
     * @param series     Input series.
     * @param windowSize Size of each window.
     * @param stride     Step between consecutive windows (1 = fully overlapping).
     * @return 2-D array of shape {@code [numWindows][windowSize]}.
     */
    public static double[][] slidingWindows(double[] series, int windowSize, int stride) {
        if (windowSize <= 0) throw new IllegalArgumentException("windowSize must be > 0");
        if (stride    <= 0) throw new IllegalArgumentException("stride must be > 0");

        int numWindows = Math.max(0, (series.length - windowSize) / stride + 1);
        double[][] result = new double[numWindows][windowSize];
        for (int i = 0; i < numWindows; i++) {
            int start = i * stride;
            System.arraycopy(series, start, result[i], 0, windowSize);
        }
        return result;
    }

    /**
     * Create lag features for a series.
     *
     * @param series Series of length {@code n}.
     * @param lags   Array of positive lag offsets.
     * @return 2-D feature matrix of shape {@code [n][lags.length]},
     *         with {@code NaN} padding for unavailable lags.
     */
    public static double[][] lagFeatures(double[] series, int[] lags) {
        int n = series.length;
        double[][] features = new double[n][lags.length];
        for (double[] row : features) Arrays.fill(row, Double.NaN);

        for (int col = 0; col < lags.length; col++) {
            int lag = lags[col];
            if (lag <= 0) throw new IllegalArgumentException("All lags must be > 0, got: " + lag);
            for (int t = lag; t < n; t++) {
                features[t][col] = series[t - lag];
            }
        }
        return features;
    }

    /**
     * Compute rolling mean over a series with a fixed window.
     * The first {@code windowSize - 1} entries are {@code NaN}.
     */
    public static double[] rollingMean(double[] series, int windowSize) {
        double[] result = new double[series.length];
        Arrays.fill(result, Double.NaN);
        double sum = 0.0;
        for (int i = 0; i < series.length; i++) {
            sum += series[i];
            if (i >= windowSize) sum -= series[i - windowSize];
            if (i >= windowSize - 1) result[i] = sum / windowSize;
        }
        return result;
    }

    /**
     * Compute rolling standard deviation over a series with a fixed window.
     * The first {@code windowSize - 1} entries are {@code NaN}.
     */
    public static double[] rollingStd(double[] series, int windowSize) {
        double[] mean   = rollingMean(series, windowSize);
        double[] result = new double[series.length];
        Arrays.fill(result, Double.NaN);

        for (int i = windowSize - 1; i < series.length; i++) {
            double m = mean[i];
            double variance = 0.0;
            for (int j = i - windowSize + 1; j <= i; j++) {
                double diff = series[j] - m;
                variance += diff * diff;
            }
            result[i] = Math.sqrt(variance / windowSize);
        }
        return result;
    }

    /**
     * Apply seasonal differencing (lag = period).
     *
     * @param series Input series.
     * @param period Seasonal period.
     * @return Seasonally differenced series of length {@code series.length - period}.
     */
    public static double[] seasonalDifference(double[] series, int period) {
        if (period <= 0) throw new IllegalArgumentException("period must be > 0");
        if (series.length <= period)
            throw new IllegalArgumentException("series too short for seasonal difference");
        double[] result = new double[series.length - period];
        for (int i = 0; i < result.length; i++) {
            result[i] = series[i + period] - series[i];
        }
        return result;
    }

    /**
     * Compute autocorrelation function (ACF) up to {@code maxLag}.
     *
     * @return Array of length {@code maxLag + 1}, where index 0 is always 1.0.
     */
    public static double[] acf(double[] series, int maxLag) {
        int n = series.length;
        double m = Arrays.stream(series).average().orElse(0.0);
        double variance = 0.0;
        for (double v : series) variance += (v - m) * (v - m);

        double[] result = new double[maxLag + 1];
        result[0] = 1.0;
        for (int lag = 1; lag <= maxLag; lag++) {
            double cov = 0.0;
            for (int t = lag; t < n; t++) {
                cov += (series[t] - m) * (series[t - lag] - m);
            }
            result[lag] = (variance > 1e-12) ? cov / variance : 0.0;
        }
        return result;
    }
}
