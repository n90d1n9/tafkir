package tech.kayys.tafkir.ml.timeseries.neural;

import tech.kayys.tafkir.ml.timeseries.api.Forecaster;
import tech.kayys.tafkir.ml.timeseries.data.MinMaxScaler;

import java.util.Arrays;
import java.util.Random;

/**
 * LSTM-based univariate time-series forecaster trained with BPTT and Adam.
 *
 * <p>Architecture:
 * <pre>
 *   Input window [lookback] → [LSTM layers] → Dense → Output [horizon]
 * </pre>
 *
 * <p>Training is done fully in Java (no Python / native dependency).
 * For large datasets or deep networks, consider exporting to ONNX and running
 * via {@code gollek-runner-timeseries-onnx}.
 *
 * <pre>{@code
 * LstmConfig cfg = LstmConfig.builder()
 *         .hiddenSize(64).numLayers(2).lookback(24)
 *         .forecastHorizon(6).epochs(100).build();
 * LstmForecaster model = new LstmForecaster(cfg);
 * model.fit(series);
 * double[] forecast = model.predict(6);
 * }</pre>
 */
public final class LstmForecaster implements Forecaster {

    private final LstmConfig config;
    private final LstmCell[] cells;   // one per layer

    // Dense output layer: W [horizon × hiddenSize], b [horizon]
    private double[] denseW;
    private double[] denseB;
    private double[] denseMW, denseMB; // Adam moments
    private double[] denseVW, denseVB;
    private int adamStep = 0;
    private static final double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;

    private double[] fittedSeries; // scaled series for warm-start predict
    private final MinMaxScaler scaler = new MinMaxScaler();

    public LstmForecaster(LstmConfig config) {
        this.config = config;
        this.cells  = new LstmCell[config.numLayers];

        // Build stacked LSTM cells
        Random rng = new Random(config.randomSeed);
        for (int l = 0; l < config.numLayers; l++) {
            int inSize = (l == 0) ? config.inputSize : config.hiddenSize;
            cells[l] = new LstmCell(inSize, config.hiddenSize, rng.nextLong());
        }

        // Xavier-init dense output layer
        int denseSz = config.forecastHorizon * config.hiddenSize;
        double denseScale = Math.sqrt(2.0 / config.hiddenSize);
        denseW  = new double[denseSz];
        denseB  = new double[config.forecastHorizon];
        for (int k = 0; k < denseSz; k++) denseW[k] = rng.nextGaussian() * denseScale;

        denseMW = new double[denseSz]; denseVW = new double[denseSz];
        denseMB = new double[config.forecastHorizon]; denseVB = new double[config.forecastHorizon];
    }

    // ── Fit ──────────────────────────────────────────────────────────────────

    @Override
    public void fit(double[] series) {
        if (series.length < config.lookback + config.forecastHorizon)
            throw new IllegalArgumentException(
                "series too short: need >= " + (config.lookback + config.forecastHorizon) + " got " + series.length);

        double[] scaled = scaler.fitTransform(series);
        fittedSeries = scaled;

        // Build windowed dataset
        int numSamples = series.length - config.lookback - config.forecastHorizon + 1;
        double[][] X = new double[numSamples][];
        double[][] Y = new double[numSamples][];
        for (int s = 0; s < numSamples; s++) {
            X[s] = Arrays.copyOfRange(scaled, s, s + config.lookback);
            Y[s] = Arrays.copyOfRange(scaled, s + config.lookback, s + config.lookback + config.forecastHorizon);
        }

        Random rng = new Random(config.randomSeed);
        for (int epoch = 0; epoch < config.epochs; epoch++) {
            // Shuffle sample indices
            int[] idx = shuffleIdx(numSamples, rng);
            double epochLoss = 0.0;
            for (int s : idx) {
                epochLoss += trainStep(X[s], Y[s]);
            }
        }
    }

    // ── Predict ──────────────────────────────────────────────────────────────

    @Override
    public double[] predict(int horizon) {
        if (fittedSeries == null) throw new IllegalStateException("Call fit() before predict()");
        if (horizon != config.forecastHorizon)
            throw new IllegalArgumentException(
                "horizon must equal config.forecastHorizon=" + config.forecastHorizon + ", got " + horizon);

        // Use last 'lookback' values of fitted (scaled) series
        int n = fittedSeries.length;
        double[] window = Arrays.copyOfRange(fittedSeries, Math.max(0, n - config.lookback), n);

        double[] scaledPred = forwardSeq(window);
        return scaler.inverseTransform(scaledPred);
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private double trainStep(double[] xSeq, double[] yTrue) {
        // Forward through each time step
        double[][][] hStates = new double[config.numLayers][config.lookback + 1][];
        double[][][] cStates = new double[config.numLayers][config.lookback + 1][];
        for (int l = 0; l < config.numLayers; l++) {
            hStates[l][0] = new double[config.hiddenSize];
            cStates[l][0] = new double[config.hiddenSize];
        }

        // Per-timestep inputs to each layer
        double[][][] layerInputsFull = new double[config.numLayers + 1][config.lookback][];
        for (int t = 0; t < config.lookback; t++) {
            layerInputsFull[0][t] = new double[]{xSeq[t]};
        }

        // Forward all layers
        for (int l = 0; l < config.numLayers; l++) {
            for (int t = 0; t < config.lookback; t++) {
                double[] xt = layerInputsFull[l][t];
                double[] h  = cells[l].forward(xt, hStates[l][t], cStates[l][t]);
                hStates[l][t + 1] = h.clone();
                cStates[l][t + 1] = cells[l].getCellState();
                if (l < config.numLayers - 1) {
                    layerInputsFull[l + 1][t] = h.clone();
                }
            }
        }

        // Final hidden state → dense → prediction
        double[] hFinal = hStates[config.numLayers - 1][config.lookback];
        double[] yPred  = denseForward(hFinal);

        // MSE loss
        double loss = 0.0;
        double[] dLoss = new double[config.forecastHorizon];
        for (int k = 0; k < config.forecastHorizon; k++) {
            double err = yPred[k] - yTrue[k];
            loss += err * err;
            dLoss[k] = 2.0 * err / config.forecastHorizon;
        }

        // Dense backward
        double[] dhFinal = denseBackward(dLoss, hFinal);

        // BPTT through LSTM layers (reverse order)
        double[] dh = dhFinal;
        double[] dc = new double[config.hiddenSize];
        for (int l = config.numLayers - 1; l >= 0; l--) {
            for (int t = config.lookback - 1; t >= 0; t--) {
                cells[l].forward(layerInputsFull[l][t], hStates[l][t], cStates[l][t]);
                double[][] grads = cells[l].backward(dh, dc);
                dh = grads[0];
                dc = grads[1];
            }
            cells[l].applyGradients(config.learningRate, config.gradientClip);
        }
        adamStep++;
        applyDenseGradients();
        return loss / config.forecastHorizon;
    }

    /** Run the full sequence and return the dense output (scaled). */
    private double[] forwardSeq(double[] xSeq) {
        double[][] h = new double[config.numLayers][];
        double[][] c = new double[config.numLayers][];
        for (int l = 0; l < config.numLayers; l++) {
            h[l] = new double[config.hiddenSize];
            c[l] = new double[config.hiddenSize];
        }
        for (int t = 0; t < xSeq.length; t++) {
            double[] input = new double[]{xSeq[t]};
            for (int l = 0; l < config.numLayers; l++) {
                h[l] = cells[l].forward(input, h[l], c[l]);
                c[l] = cells[l].getCellState();
                input = h[l];
            }
        }
        return denseForward(h[config.numLayers - 1]);
    }

    // ── Dense layer ───────────────────────────────────────────────────────────

    private double[] denseForward(double[] h) {
        double[] out = denseB.clone();
        for (int row = 0; row < config.forecastHorizon; row++) {
            int off = row * config.hiddenSize;
            for (int col = 0; col < config.hiddenSize; col++)
                out[row] += denseW[off + col] * h[col];
        }
        return out;
    }

    private double[] dDenseW = null;
    private double[] dDenseB = null;

    private double[] denseBackward(double[] dOut, double[] h) {
        dDenseW = new double[config.forecastHorizon * config.hiddenSize];
        dDenseB = dOut.clone();
        double[] dh = new double[config.hiddenSize];
        for (int row = 0; row < config.forecastHorizon; row++) {
            int off = row * config.hiddenSize;
            for (int col = 0; col < config.hiddenSize; col++) {
                dDenseW[off + col] += dOut[row] * h[col];
                dh[col]            += dOut[row] * denseW[off + col];
            }
        }
        return dh;
    }

    private void applyDenseGradients() {
        if (dDenseW == null) return;
        double bc1 = 1 - Math.pow(BETA1, adamStep);
        double bc2 = 1 - Math.pow(BETA2, adamStep);
        for (int k = 0; k < denseW.length; k++) {
            denseMW[k] = BETA1 * denseMW[k] + (1 - BETA1) * dDenseW[k];
            denseVW[k] = BETA2 * denseVW[k] + (1 - BETA2) * dDenseW[k] * dDenseW[k];
            denseW[k]  -= config.learningRate * (denseMW[k] / bc1) / (Math.sqrt(denseVW[k] / bc2) + EPS);
        }
        for (int k = 0; k < denseB.length; k++) {
            denseMB[k] = BETA1 * denseMB[k] + (1 - BETA1) * dDenseB[k];
            denseVB[k] = BETA2 * denseVB[k] + (1 - BETA2) * dDenseB[k] * dDenseB[k];
            denseB[k]  -= config.learningRate * (denseMB[k] / bc1) / (Math.sqrt(denseVB[k] / bc2) + EPS);
        }
        dDenseW = null; dDenseB = null;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private static int[] shuffleIdx(int n, Random rng) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        return idx;
    }

    @Override
    public String name() { return "LstmForecaster(" + config.hiddenSize + "×" + config.numLayers + ")"; }
    public LstmConfig getConfig() { return config; }
}
