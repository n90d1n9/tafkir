package tech.kayys.tafkir.ml.timeseries.statistical;

import tech.kayys.tafkir.ml.timeseries.api.Forecaster;

import java.util.Arrays;

/**
 * ARIMA(p, d, q) forecaster.
 *
 * <p><b>Estimation strategy</b>
 * <ul>
 *   <li>AR coefficients: Yule-Walker equations (exact for Gaussian series).</li>
 *   <li>MA coefficients: ACF-moment matching on the AR residuals (fast approximation).</li>
 *   <li>Differencing inversion: cumulative sum from stored in-sample tail values.</li>
 * </ul>
 *
 * <p><b>Limitations</b>: For production use with large {@code q} consider replacing the
 * MA estimator with conditional maximum likelihood (CSS). The Yule-Walker estimator
 * can be biased for short series; use at least 50 observations per order of magnitude.
 *
 * <pre>{@code
 * Forecaster arima = new ArimaForecaster(2, 1, 1);
 * arima.fit(series);
 * double[] forecasts = arima.predict(12);
 * }</pre>
 */
public final class ArimaForecaster implements Forecaster {

    private final int p;
    private final int d;
    private final int q;

    private double[] arCoef;           // AR coefficients φ₁…φₚ
    private double[] maCoef;           // MA coefficients θ₁…θq
    private double[] differencedSeries; // series after d differencing steps
    private double[][] levelTails;      // last few values at each differencing level for inversion
    private double[] residuals;         // in-sample ARMA residuals

    public ArimaForecaster(int p, int d, int q) {
        if (p < 0 || d < 0 || q < 0)
            throw new IllegalArgumentException("ARIMA orders must be non-negative; got p=" + p + " d=" + d + " q=" + q);
        this.p = p; this.d = d; this.q = q;
    }

    // ── Fit ──────────────────────────────────────────────────────────────────

    @Override
    public void fit(double[] series) {
        if (series == null || series.length < Math.max(1, p + d + q + 1))
            throw new IllegalArgumentException("series too short for ARIMA(" + p + "," + d + "," + q + ")");

        // 1. Differencing
        levelTails = new double[d][];
        double[] current = series.clone();
        for (int i = 0; i < d; i++) {
            levelTails[i] = current.clone(); // store for inversion
            current = difference(current);
        }
        differencedSeries = current;

        // 2. AR estimation via Yule-Walker
        arCoef = (p > 0) ? yuleWalker(differencedSeries, p) : new double[0];

        // 3. MA estimation via ACF of AR residuals
        double[] arResiduals = arResiduals(differencedSeries);
        maCoef = (q > 0) ? estimateMa(arResiduals, q) : new double[0];

        // 4. Compute in-sample ARMA residuals for MA forecasting
        residuals = armaResiduals(differencedSeries);
    }

    // ── Predict ──────────────────────────────────────────────────────────────

    @Override
    public double[] predict(int horizon) {
        if (differencedSeries == null)
            throw new IllegalStateException("Call fit() before predict()");

        int n = differencedSeries.length;

        // Extended series and residuals (future residuals assumed 0)
        double[] ext = Arrays.copyOf(differencedSeries, n + horizon);
        double[] extRes = Arrays.copyOf(residuals, n + horizon); // tail already zero-padded

        for (int h = 0; h < horizon; h++) {
            int t = n + h;
            double val = 0.0;
            for (int i = 0; i < p; i++) {
                if (t - i - 1 >= 0) val += arCoef[i] * ext[t - i - 1];
            }
            // MA: only in-sample residuals available (future = 0)
            for (int j = 0; j < q; j++) {
                int resIdx = t - j - 1;
                if (resIdx >= 0 && resIdx < n) val += maCoef[j] * extRes[resIdx];
            }
            ext[t] = val;
        }

        // Extract raw differenced forecasts
        double[] forecasts = Arrays.copyOfRange(ext, n, n + horizon);

        // Invert differencing level by level (reverse order)
        for (int i = d - 1; i >= 0; i--) {
            forecasts = undifference(forecasts, levelTails[i]);
        }
        return forecasts;
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private static double[] difference(double[] series) {
        double[] result = new double[series.length - 1];
        for (int i = 0; i < result.length; i++) result[i] = series[i + 1] - series[i];
        return result;
    }

    /** Invert one level of differencing using the tail of the pre-differenced series. */
    private static double[] undifference(double[] diffs, double[] prevLevel) {
        double lastKnown = prevLevel[prevLevel.length - 1];
        double[] result = new double[diffs.length];
        result[0] = lastKnown + diffs[0];
        for (int i = 1; i < diffs.length; i++) result[i] = result[i - 1] + diffs[i];
        return result;
    }

    /** Yule-Walker: solve Toeplitz system R·φ = r for AR coefficients. */
    private static double[] yuleWalker(double[] series, int order) {
        double[] acf = acf(series, order);
        // Build Toeplitz matrix
        double[][] R = new double[order][order];
        for (int i = 0; i < order; i++)
            for (int j = 0; j < order; j++)
                R[i][j] = acf[Math.abs(i - j)];
        double[] rhs = Arrays.copyOfRange(acf, 1, order + 1);
        return solveLinear(R, rhs);
    }

    /** ACF-moment MA estimation on residuals. Fast but approximate. */
    private static double[] estimateMa(double[] residuals, int qOrder) {
        double[] acf = acf(residuals, qOrder);
        // θ̂_j ≈ acf[j] / (1 + Σ acf[k]²)
        double denom = 1.0;
        for (int k = 1; k <= qOrder; k++) denom += acf[k] * acf[k];
        double[] ma = new double[qOrder];
        for (int j = 0; j < qOrder; j++) ma[j] = acf[j + 1] / denom;
        return ma;
    }

    private double[] arResiduals(double[] series) {
        int n = series.length;
        double[] res = new double[n];
        for (int t = p; t < n; t++) {
            double pred = 0.0;
            for (int i = 0; i < p; i++) pred += arCoef[i] * series[t - i - 1];
            res[t] = series[t] - pred;
        }
        return res;
    }

    private double[] armaResiduals(double[] series) {
        int n = series.length;
        int start = Math.max(p, q);
        double[] res = new double[n];
        for (int t = start; t < n; t++) {
            double pred = 0.0;
            for (int i = 0; i < p; i++) pred += arCoef[i] * series[t - i - 1];
            for (int j = 0; j < q; j++) {
                if (t - j - 1 >= 0) pred += maCoef[j] * res[t - j - 1];
            }
            res[t] = series[t] - pred;
        }
        return res;
    }

    /** Compute autocorrelation function up to maxLag. Index 0 is always 1.0. */
    private static double[] acf(double[] series, int maxLag) {
        int n = series.length;
        double mean = 0.0;
        for (double v : series) mean += v;
        mean /= n;

        double variance = 0.0;
        for (double v : series) variance += (v - mean) * (v - mean);

        double[] result = new double[maxLag + 1];
        result[0] = 1.0;
        for (int lag = 1; lag <= maxLag; lag++) {
            double cov = 0.0;
            for (int t = lag; t < n; t++) cov += (series[t] - mean) * (series[t - lag] - mean);
            result[lag] = (variance > 1e-12) ? cov / variance : 0.0;
        }
        return result;
    }

    /** Gaussian elimination with partial pivoting. */
    private static double[] solveLinear(double[][] A, double[] b) {
        int n = b.length;
        if (n == 0) return new double[0];
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
            double[] tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp;

            if (Math.abs(aug[col][col]) < 1e-12) continue;
            for (int row = col + 1; row < n; row++) {
                double factor = aug[row][col] / aug[col][col];
                for (int k = col; k <= n; k++) aug[row][k] -= factor * aug[col][k];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
            if (Math.abs(aug[i][i]) > 1e-12) x[i] /= aug[i][i];
        }
        return x;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    @Override public String name() { return "ARIMA(" + p + "," + d + "," + q + ")"; }
    public int getP() { return p; }
    public int getD() { return d; }
    public int getQ() { return q; }
    public double[] getArCoefficients() { return (arCoef == null) ? null : arCoef.clone(); }
    public double[] getMaCoefficients() { return (maCoef == null) ? null : maCoef.clone(); }
}
