package tech.kayys.tafkir.ml.timeseries;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.ml.timeseries.api.ForecastRequest;
import tech.kayys.tafkir.ml.timeseries.api.ForecastResult;
import tech.kayys.tafkir.ml.timeseries.metrics.ForecastMetrics;
import tech.kayys.tafkir.ml.timeseries.neural.LstmConfig;
import tech.kayys.tafkir.ml.timeseries.neural.LstmForecaster;
import tech.kayys.tafkir.ml.timeseries.statistical.ArimaForecaster;
import tech.kayys.tafkir.ml.timeseries.statistical.ExponentialSmoothing;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Benchmark test suite using canonical time-series datasets (Classic Box-Jenkins Airline Passengers).
 * Compares ARIMA, Exponential Smoothing (Holt-Winters), and LSTM deep learning models.
 */
public class BenchmarkTimeSeriesTest {

    // Classic International Airline Passengers dataset (Monthly totals in thousands, 1949-1960, 144 observations)
    private static final double[] AIRLINE_PASSENGERS = {
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    };

    private static final int HORIZON = 12; // 1-year forecast horizon
    private static final int TRAIN_LEN = AIRLINE_PASSENGERS.length - HORIZON;

    @Test
    @DisplayName("Benchmark ARIMA(1,1,1) on Airline Passengers dataset")
    void testArimaOnAirlineData() {
        double[] train = Arrays.copyOfRange(AIRLINE_PASSENGERS, 0, TRAIN_LEN);
        double[] actual = Arrays.copyOfRange(AIRLINE_PASSENGERS, TRAIN_LEN, AIRLINE_PASSENGERS.length);

        ArimaForecaster arima = new ArimaForecaster(1, 1, 1);
        arima.fit(train);
        double[] forecast = arima.predict(HORIZON);

        assertEquals(HORIZON, forecast.length);
        for (double v : forecast) {
            assertFalse(Double.isNaN(v), "ARIMA prediction contains NaN");
            assertTrue(v > 0, "Passenger count should be positive");
        }

        ForecastMetrics.MetricSummary metrics = ForecastMetrics.summary(actual, forecast, train);
        System.out.println("ARIMA(1,1,1) Benchmark Results: " + metrics);
        assertTrue(metrics.mape() < 30.0, "MAPE should be reasonable (< 30%), got: " + metrics.mape());
    }

    @Test
    @DisplayName("Benchmark Holt-Winters Multi-Seasonal on Airline Passengers dataset")
    void testHoltWintersOnAirlineData() {
        double[] train = Arrays.copyOfRange(AIRLINE_PASSENGERS, 0, TRAIN_LEN);
        double[] actual = Arrays.copyOfRange(AIRLINE_PASSENGERS, TRAIN_LEN, AIRLINE_PASSENGERS.length);

        // Period = 12 (monthly seasonality)
        ExponentialSmoothing hw = ExponentialSmoothing.holtWinters(0.4, 0.1, 0.3, 12);
        hw.fit(train);
        double[] forecast = hw.predict(HORIZON);

        assertEquals(HORIZON, forecast.length);
        for (double v : forecast) {
            assertFalse(Double.isNaN(v));
            assertTrue(v > 100);
        }

        ForecastMetrics.MetricSummary metrics = ForecastMetrics.summary(actual, forecast, train);
        System.out.println("Holt-Winters Benchmark Results: " + metrics);
        assertTrue(metrics.mape() < 25.0, "Holt-Winters MAPE should be < 25%, got: " + metrics.mape());
    }

    @Test
    @DisplayName("Benchmark LSTM Forecaster on Airline Passengers dataset")
    void testLstmOnAirlineData() {
        double[] train = Arrays.copyOfRange(AIRLINE_PASSENGERS, 0, TRAIN_LEN);
        double[] actual = Arrays.copyOfRange(AIRLINE_PASSENGERS, TRAIN_LEN, AIRLINE_PASSENGERS.length);

        LstmConfig config = LstmConfig.builder()
                .inputSize(1)
                .hiddenSize(32)
                .numLayers(1)
                .lookback(24) // 2 years lookback
                .forecastHorizon(HORIZON)
                .learningRate(0.01)
                .epochs(80)
                .randomSeed(42L)
                .build();

        LstmForecaster lstm = new LstmForecaster(config);
        ForecastResult result = lstm.forecast(ForecastRequest.of(train, HORIZON));

        assertEquals(HORIZON, result.horizon());
        for (double v : result.values()) {
            assertFalse(Double.isNaN(v));
            assertTrue(v > 0);
        }

        ForecastMetrics.MetricSummary metrics = ForecastMetrics.summary(actual, result.values(), train);
        System.out.println("LSTM Benchmark Results: " + metrics);
        assertFalse(Double.isNaN(metrics.rmse()));
    }
}
