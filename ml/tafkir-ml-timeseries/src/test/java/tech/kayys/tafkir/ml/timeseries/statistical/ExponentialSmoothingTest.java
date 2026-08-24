package tech.kayys.tafkir.ml.timeseries.statistical;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ExponentialSmoothingTest {

    @Test
    void simple_constantSeries_returnsSameValue() {
        double[] series = new double[30];
        java.util.Arrays.fill(series, 5.0);
        ExponentialSmoothing ses = ExponentialSmoothing.simple(0.3);
        ses.fit(series);
        double[] forecast = ses.predict(3);
        for (double v : forecast) assertEquals(5.0, v, 1e-6);
    }

    @Test
    void holt_linearSeries_followsTrend() {
        double[] series = new double[30];
        for (int i = 0; i < 30; i++) series[i] = 2.0 * i + 1.0;
        ExponentialSmoothing holt = ExponentialSmoothing.holt(0.8, 0.8);
        holt.fit(series);
        double[] forecast = holt.predict(5);
        // The forecast should be roughly linear and growing
        assertTrue(forecast[4] > forecast[0], "Holt forecast should grow on upward trend");
    }

    @Test
    void holtWinters_seasonal_nonNaN() {
        // Monthly sales data: 24 months of a seasonal pattern
        double[] series = new double[24];
        for (int i = 0; i < 24; i++) {
            series[i] = 100 + 20 * Math.sin(2 * Math.PI * i / 12) + i;
        }
        ExponentialSmoothing hw = ExponentialSmoothing.holtWinters(0.3, 0.1, 0.2, 12);
        hw.fit(series);
        double[] forecast = hw.predict(12);
        assertEquals(12, forecast.length);
        for (double v : forecast) assertFalse(Double.isNaN(v));
    }

    @Test
    void simpleFactory_works() {
        ExponentialSmoothing ses = ExponentialSmoothing.simple(0.5);
        ses.fit(new double[]{1, 2, 3, 4, 5});
        double[] f = ses.predict(3);
        assertEquals(3, f.length);
        // All values should be the same (level only)
        assertEquals(f[0], f[1], 1e-10);
        assertEquals(f[1], f[2], 1e-10);
    }

    @Test
    void invalidAlpha_throws() {
        assertThrows(IllegalArgumentException.class, () -> new ExponentialSmoothing(
                ExponentialSmoothing.Mode.SIMPLE, 1.5, 0, 0, 0));
    }

    @Test
    void holtWintersNeedsPeriod_throws() {
        assertThrows(IllegalArgumentException.class, () -> new ExponentialSmoothing(
                ExponentialSmoothing.Mode.HOLT_WINTERS, 0.3, 0.1, 0.2, 1));
    }
}
