package tech.kayys.tafkir.ml.timeseries.neural;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LstmForecasterTest {

    /** Simple sine wave series for sanity checks. */
    private static double[] sineWave(int n) {
        double[] s = new double[n];
        for (int i = 0; i < n; i++) s[i] = Math.sin(2 * Math.PI * i / 12.0);
        return s;
    }

    @Test
    void predict_returnsCorrectLength() {
        LstmConfig cfg = LstmConfig.builder()
                .inputSize(1).hiddenSize(8).numLayers(1)
                .lookback(12).forecastHorizon(3)
                .epochs(2).build();
        LstmForecaster model = new LstmForecaster(cfg);
        model.fit(sineWave(60));
        double[] forecast = model.predict(3);
        assertEquals(3, forecast.length);
    }

    @Test
    void predict_noNaN() {
        LstmConfig cfg = LstmConfig.builder()
                .inputSize(1).hiddenSize(8).numLayers(1)
                .lookback(12).forecastHorizon(3)
                .epochs(5).build();
        LstmForecaster model = new LstmForecaster(cfg);
        model.fit(sineWave(60));
        for (double v : model.predict(3)) assertFalse(Double.isNaN(v), "NaN in forecast");
    }

    @Test
    void stacked_lstm_works() {
        LstmConfig cfg = LstmConfig.builder()
                .inputSize(1).hiddenSize(16).numLayers(2)
                .lookback(12).forecastHorizon(3)
                .epochs(3).build();
        LstmForecaster model = new LstmForecaster(cfg);
        model.fit(sineWave(80));
        assertEquals(3, model.predict(3).length);
    }

    @Test
    void wrongHorizon_throws() {
        LstmConfig cfg = LstmConfig.builder()
                .forecastHorizon(6).lookback(10).epochs(1).build();
        LstmForecaster model = new LstmForecaster(cfg);
        model.fit(sineWave(60));
        assertThrows(IllegalArgumentException.class, () -> model.predict(3));
    }

    @Test
    void notFitted_throws() {
        LstmConfig cfg = LstmConfig.builder().forecastHorizon(3).lookback(10).epochs(1).build();
        assertThrows(IllegalStateException.class, () -> new LstmForecaster(cfg).predict(3));
    }

    @Test
    void name_contains_config() {
        LstmConfig cfg = LstmConfig.builder().hiddenSize(32).numLayers(2).forecastHorizon(3).lookback(10).build();
        LstmForecaster model = new LstmForecaster(cfg);
        assertTrue(model.name().contains("32"));
        assertTrue(model.name().contains("2"));
    }
}
