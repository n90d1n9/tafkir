package tech.kayys.tafkir.ml.timeseries.metrics;

import java.util.Objects;

/**
 * Standard point-forecast accuracy metrics.
 *
 * <p>All methods accept parallel arrays {@code actual} and {@code predicted} of equal length.
 *
 * <ul>
 *   <li>{@link #mae(double[], double[])} — Mean Absolute Error</li>
 *   <li>{@link #rmse(double[], double[])} — Root Mean Squared Error</li>
 *   <li>{@link #mape(double[], double[])} — Mean Absolute Percentage Error (%)</li>
 *   <li>{@link #smape(double[], double[])} — Symmetric MAPE (%)</li>
 *   <li>{@link #mase(double[], double[], double[])} — Mean Absolute Scaled Error</li>
 *   <li>{@link #wape(double[], double[])} — Weighted Absolute Percentage Error</li>
 *   <li>{@link #r2(double[], double[])} — Coefficient of Determination (R²)</li>
 * </ul>
 */
public final class ForecastMetrics {

    private ForecastMetrics() {}

    // ── Validation ────────────────────────────────────────────────────────────

    private static void validate(double[] actual, double[] predicted) {
        Objects.requireNonNull(actual,    "actual must not be null");
        Objects.requireNonNull(predicted, "predicted must not be null");
        if (actual.length != predicted.length)
            throw new IllegalArgumentException(
                "actual.length (" + actual.length + ") != predicted.length (" + predicted.length + ")");
        if (actual.length == 0)
            throw new IllegalArgumentException("Arrays must not be empty");
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    /**
     * Mean Absolute Error: {@code mean(|actual - predicted|)}.
     */
    public static double mae(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - predicted[i]);
        return sum / actual.length;
    }

    /**
     * Mean Squared Error: {@code mean((actual - predicted)^2)}.
     */
    public static double mse(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double d = actual[i] - predicted[i];
            sum += d * d;
        }
        return sum / actual.length;
    }

    /**
     * Root Mean Squared Error: {@code sqrt(MSE)}.
     */
    public static double rmse(double[] actual, double[] predicted) {
        return Math.sqrt(mse(actual, predicted));
    }

    /**
     * Mean Absolute Percentage Error (%).
     * Undefined (returns {@code Double.NaN}) when any {@code actual[i] == 0}.
     */
    public static double mape(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.abs(actual[i]) < 1e-12) return Double.NaN;
            sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
        }
        return 100.0 * sum / actual.length;
    }

    /**
     * Symmetric Mean Absolute Percentage Error (%).
     * {@code 100 * mean(2|actual - predicted| / (|actual| + |predicted|))}
     */
    public static double smape(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double denom = Math.abs(actual[i]) + Math.abs(predicted[i]);
            sum += (denom < 1e-12) ? 0.0 : 2.0 * Math.abs(actual[i] - predicted[i]) / denom;
        }
        return 100.0 * sum / actual.length;
    }

    /**
     * Mean Absolute Scaled Error.
     * Scale = MAE of the in-sample naïve (random-walk) forecast on {@code trainSeries}.
     *
     * @param actual      Out-of-sample actuals.
     * @param predicted   Out-of-sample predictions.
     * @param trainSeries In-sample training series used to compute the naïve scale.
     */
    public static double mase(double[] actual, double[] predicted, double[] trainSeries) {
        validate(actual, predicted);
        Objects.requireNonNull(trainSeries, "trainSeries must not be null");
        if (trainSeries.length < 2) throw new IllegalArgumentException("trainSeries must have >= 2 points");

        // Naïve MAE on training set (one-step random walk)
        double naiveSum = 0.0;
        for (int t = 1; t < trainSeries.length; t++) {
            naiveSum += Math.abs(trainSeries[t] - trainSeries[t - 1]);
        }
        double scale = naiveSum / (trainSeries.length - 1);
        if (scale < 1e-12) return Double.NaN;

        return mae(actual, predicted) / scale;
    }

    /**
     * Weighted Absolute Percentage Error (WAPE / nMAE).
     * {@code sum(|actual - predicted|) / sum(|actual|)}
     */
    public static double wape(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < actual.length; i++) {
            numerator   += Math.abs(actual[i] - predicted[i]);
            denominator += Math.abs(actual[i]);
        }
        return (denominator < 1e-12) ? Double.NaN : 100.0 * numerator / denominator;
    }

    /**
     * Coefficient of Determination (R²).
     * 1.0 = perfect forecast; can be negative for very poor models.
     */
    public static double r2(double[] actual, double[] predicted) {
        validate(actual, predicted);
        double mean = 0.0;
        for (double v : actual) mean += v;
        mean /= actual.length;

        double ssTot = 0.0, ssRes = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double d = actual[i] - mean;
            ssTot += d * d;
            double r = actual[i] - predicted[i];
            ssRes += r * r;
        }
        return (ssTot < 1e-12) ? 1.0 : 1.0 - ssRes / ssTot;
    }

    /**
     * Compute a full metric summary record for convenience.
     *
     * @param trainSeries Required for MASE scale. Pass {@code null} to skip MASE (NaN).
     */
    public static MetricSummary summary(double[] actual, double[] predicted, double[] trainSeries) {
        double maseValue = (trainSeries != null) ? mase(actual, predicted, trainSeries) : Double.NaN;
        return new MetricSummary(
                mae(actual, predicted),
                rmse(actual, predicted),
                mape(actual, predicted),
                smape(actual, predicted),
                maseValue,
                wape(actual, predicted),
                r2(actual, predicted)
        );
    }

    // ── Result record ─────────────────────────────────────────────────────────

    /**
     * Snapshot of common accuracy metrics.
     */
    public record MetricSummary(
            double mae,
            double rmse,
            double mape,
            double smape,
            double mase,
            double wape,
            double r2
    ) {
        @Override
        public String toString() {
            return String.format(
                    "MetricSummary[MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, sMAPE=%.2f%%, MASE=%.4f, WAPE=%.2f%%, R²=%.4f]",
                    mae, rmse, mape, smape, mase, wape, r2);
        }
    }
}
