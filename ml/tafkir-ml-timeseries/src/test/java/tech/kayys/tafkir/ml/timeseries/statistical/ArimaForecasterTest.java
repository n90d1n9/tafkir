package tech.kayys.tafkir.ml.timeseries.statistical;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ArimaForecasterTest {

    /** Generate a simple AR(1) process: y_t = 0.7*y_{t-1} + noise (no noise for determinism). */
    private static double[] ar1Series(int n, double phi) {
        double[] s = new double[n];
        s[0] = 1.0;
        for (int t = 1; t < n; t++) s[t] = phi * s[t - 1];
        return s;
    }

    @Test
    void predict_ar1_returnsNonNaN() {
        double[] series = ar1Series(100, 0.7);
        ArimaForecaster arima = new ArimaForecaster(1, 0, 0);
        arima.fit(series);
        double[] forecast = arima.predict(5);
        assertEquals(5, forecast.length);
        for (double v : forecast) assertFalse(Double.isNaN(v), "NaN in forecast");
    }

    @Test
    void ar1_forecastDecays() {
        double[] series = ar1Series(100, 0.7);
        ArimaForecaster arima = new ArimaForecaster(1, 0, 0);
        arima.fit(series);
        double[] forecast = arima.predict(5);
        // AR(1) with phi<1 should produce decaying forecasts toward 0
        assertTrue(Math.abs(forecast[4]) < Math.abs(forecast[0]),
                "Forecast should decay for stationary AR(1)");
    }

    @Test
    void predict_arima011_nonNaN() {
        // ARIMA(0,1,1) is equivalent to SES
        double[] series = new double[80];
        for (int i = 0; i < 80; i++) series[i] = Math.sin(i * 0.3) + i * 0.05;
        ArimaForecaster arima = new ArimaForecaster(0, 1, 1);
        arima.fit(series);
        double[] forecast = arima.predict(6);
        assertEquals(6, forecast.length);
        for (double v : forecast) assertFalse(Double.isNaN(v));
    }

    @Test
    void invalidOrders_throw() {
        assertThrows(IllegalArgumentException.class, () -> new ArimaForecaster(-1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new ArimaForecaster(0, -1, 0));
    }

    @Test
    void notFitted_throws() {
        ArimaForecaster arima = new ArimaForecaster(1, 0, 0);
        assertThrows(IllegalStateException.class, () -> arima.predict(5));
    }

    @Test
    void name_matches_orders() {
        assertEquals("ARIMA(2,1,2)", new ArimaForecaster(2, 1, 2).name());
    }
}
