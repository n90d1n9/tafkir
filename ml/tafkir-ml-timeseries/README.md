# tafkir-ml-timeseries

Time-series forecasting library for the **Tafkir** ML framework.

## Overview

Provides a unified `Forecaster` API covering three categories of models:

| Category | Models | GPU? |
|---|---|---|
| **Statistical** | ARIMA, Exponential Smoothing (SES / Holt / Holt-Winters) | ❌ Pure Java |
| **Neural** | LSTM (pure Java BPTT + Adam) | ❌ CPU only |
| **Pre-trained (inference)** | Any ONNX/SafeTensor time-series model via Gollek | ✅ Metal / CUDA |

---

## Quick Start

### ARIMA(2,1,1)
```java
import tech.kayys.tafkir.ml.timeseries.statistical.ArimaForecaster;

double[] series = loadYourSeries();  // univariate, e.g. monthly sales

ArimaForecaster arima = new ArimaForecaster(2, 1, 1);
arima.fit(series);
double[] forecast = arima.predict(12);   // 12-step ahead forecast
```

### Exponential Smoothing (Holt-Winters Additive)
```java
import tech.kayys.tafkir.ml.timeseries.statistical.ExponentialSmoothing;

// alpha=0.3 (level), beta=0.1 (trend), gamma=0.2 (seasonal), period=12
ExponentialSmoothing hw = ExponentialSmoothing.holtWinters(0.3, 0.1, 0.2, 12);
hw.fit(series);
double[] forecast = hw.predict(12);
```

### LSTM Forecaster
```java
import tech.kayys.tafkir.ml.timeseries.neural.*;

LstmConfig cfg = LstmConfig.builder()
        .inputSize(1)          // univariate
        .hiddenSize(64)        // LSTM state size
        .numLayers(2)          // stacked LSTM layers
        .lookback(24)          // input window length
        .forecastHorizon(6)    // output steps
        .epochs(100)
        .learningRate(1e-3)
        .build();

LstmForecaster lstm = new LstmForecaster(cfg);
lstm.fit(series);
double[] forecast = lstm.predict(6);
```

### Forecast Metrics
```java
import tech.kayys.tafkir.ml.timeseries.metrics.ForecastMetrics;

double mae   = ForecastMetrics.mae(actual, predicted);
double rmse  = ForecastMetrics.rmse(actual, predicted);
double smape = ForecastMetrics.smape(actual, predicted);
double mase  = ForecastMetrics.mase(actual, predicted, trainSeries);

// Or get all at once:
ForecastMetrics.MetricSummary metrics = ForecastMetrics.summary(actual, predicted, trainSeries);
System.out.println(metrics);
// MetricSummary[MAE=0.0421, RMSE=0.0563, MAPE=2.31%, sMAPE=2.28%, MASE=0.88, WAPE=2.31%, R²=0.9923]
```

### Data Utilities
```java
import tech.kayys.tafkir.ml.timeseries.data.*;

// Min-max scaling
MinMaxScaler scaler = new MinMaxScaler();
double[] scaled = scaler.fitTransform(series);
double[] original = scaler.inverseTransform(scaled);

// Sliding windows
double[][] windows = TimeSeriesWindow.slidingWindows(series, 24 /*size*/, 1 /*stride*/);

// Lag features
double[][] features = TimeSeriesWindow.lagFeatures(series, new int[]{1, 7, 14, 28});

// ACF
double[] acf = TimeSeriesWindow.acf(series, 20 /*maxLag*/);

// Supervised dataset (lookback=24, horizon=6)
TimeSeriesDataset ds = new TimeSeriesDataset(series, 24, 6);
for (int i = 0; i < ds.size(); i++) {
    double[] x = ds.getInput(i);   // [24]
    double[] y = ds.getTarget(i);  // [6]
}
```

---

## Model Selection Guide

| Scenario | Recommended Model |
|---|---|
| No obvious trend/season, short series | `ArimaForecaster(1,0,0)` |
| Clear trend | `ExponentialSmoothing.holt(0.4, 0.2)` |
| Clear trend + seasonality | `ExponentialSmoothing.holtWinters(0.3, 0.1, 0.2, 12)` |
| Non-linear patterns, ≥200 observations | `LstmForecaster` |
| Pre-trained model from HuggingFace | `gollek pull hf:amazon/chronos-t5-small` → `gollek run` |

### Serving Pre-trained Models via Gollek
```bash
# Pull a pre-trained time-series model (auto-tagged as 'timeseries' in gollek list)
gollek pull hf:amazon/chronos-t5-small

# List — shows TASK = timeseries in yellow
gollek list

# Run via REST
gollek serve --port 8080
```

---

## Package Layout

```
tech.kayys.tafkir.ml.timeseries
  .api                   Forecaster SPI, ForecastRequest/Result, TimeSeriesDataset
  .data                  MinMaxScaler, StandardScaler, TimeSeriesWindow
  .metrics               ForecastMetrics (MAE, RMSE, MAPE, sMAPE, MASE, WAPE, R²)
  .statistical           ArimaForecaster, ExponentialSmoothing (SES/Holt/HW)
  .neural                LstmConfig, LstmCell, LstmForecaster
```

---

## Gradle Dependency

```kotlin
implementation("tech.kayys.tafkir:tafkir-ml-timeseries:0.1.0-SNAPSHOT")
```

---

## Roadmap

- [ ] Auto-ARIMA (order selection via AIC/BIC)
- [ ] Prophet-style additive model (trend + seasonality + holidays)
- [ ] Temporal Fusion Transformer (TFT)
- [ ] N-BEATS / N-HiTS
- [ ] ONNX runner SPI for HuggingFace Chronos / TimesFM
- [ ] Multivariate input support
- [ ] Prediction intervals / quantile forecasts
