package tech.kayys.tafkir.ml.timeseries.metrics;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ForecastMetricsTest {

    private static final double DELTA = 1e-6;

    @Test
    void mae_perfectForecast() {
        double[] a = {1, 2, 3, 4, 5};
        assertEquals(0.0, ForecastMetrics.mae(a, a), DELTA);
    }

    @Test
    void mae_knownValues() {
        double[] actual    = {3.0, 5.0, 4.0, 7.0, 2.0};
        double[] predicted = {2.5, 5.5, 3.5, 8.0, 2.0};
        // |0.5| + |0.5| + |0.5| + |1| + |0| = 2.5 / 5 = 0.5
        assertEquals(0.5, ForecastMetrics.mae(actual, predicted), DELTA);
    }

    @Test
    void rmse_knownValues() {
        double[] actual    = {1.0, 2.0, 3.0};
        double[] predicted = {2.0, 2.0, 2.0};
        // errors: 1, 0, 1  → MSE = 2/3 → RMSE = sqrt(2/3)
        assertEquals(Math.sqrt(2.0 / 3.0), ForecastMetrics.rmse(actual, predicted), DELTA);
    }

    @Test
    void smape_symmetric() {
        double[] actual    = {100.0};
        double[] predicted = {110.0};
        // 2 * 10 / 210 * 100 = 9.5238...%
        assertEquals(100.0 * 2.0 * 10.0 / 210.0, ForecastMetrics.smape(actual, predicted), 1e-4);
    }

    @Test
    void mase_randomWalkBaseline() {
        // If we predict the mean of train set on a random walk series,
        // MASE should be roughly 1 (scale = naive error)
        double[] train = new double[100];
        for (int i = 0; i < 100; i++) train[i] = i;   // linear — naive = 1 each step
        double[] actual    = {100.0, 101.0, 102.0};
        double[] predicted = {100.0, 101.0, 102.0};   // perfect
        assertEquals(0.0, ForecastMetrics.mase(actual, predicted, train), DELTA);
    }

    @Test
    void r2_perfectFit() {
        double[] a = {1, 2, 3, 4, 5};
        assertEquals(1.0, ForecastMetrics.r2(a, a), DELTA);
    }

    @Test
    void r2_worseThanMean() {
        double[] actual    = {1.0, 2.0, 3.0};
        double[] predicted = {10.0, 0.0, -5.0};
        assertTrue(ForecastMetrics.r2(actual, predicted) < 0);
    }

    @Test
    void summary_noNaN() {
        double[] a = {1.0, 2.0, 3.0};
        double[] p = {1.1, 2.2, 2.9};
        ForecastMetrics.MetricSummary s = ForecastMetrics.summary(a, p, a);
        assertFalse(Double.isNaN(s.mae()));
        assertFalse(Double.isNaN(s.rmse()));
        assertFalse(Double.isNaN(s.smape()));
        assertFalse(Double.isNaN(s.r2()));
    }

    @Test
    void lengthMismatch_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ForecastMetrics.mae(new double[]{1}, new double[]{1, 2}));
    }
}
