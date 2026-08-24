package tech.kayys.tafkir.ml.timeseries.neural;

/**
 * Hyperparameter configuration for an LSTM-based forecaster.
 *
 * <p>All fields have sensible defaults accessible via {@link Builder}.
 *
 * <pre>{@code
 * LstmConfig cfg = LstmConfig.builder()
 *     .inputSize(1)
 *     .hiddenSize(64)
 *     .numLayers(2)
 *     .forecastHorizon(12)
 *     .epochs(100)
 *     .learningRate(1e-3)
 *     .build();
 * }</pre>
 */
public final class LstmConfig {

    // Architecture
    public final int    inputSize;       // number of input features (1 for univariate)
    public final int    hiddenSize;      // LSTM hidden dimension
    public final int    numLayers;       // number of stacked LSTM layers
    public final int    forecastHorizon; // output steps

    // Training
    public final int    epochs;          // maximum training epochs
    public final double learningRate;    // SGD/Adam learning rate
    public final double dropout;         // dropout probability between LSTM layers (0 = disabled)
    public final int    lookback;        // input window length
    public final double gradientClip;    // max L2 norm for gradient clipping (0 = disabled)
    public final long   randomSeed;      // for reproducible weight initialisation

    private LstmConfig(Builder b) {
        this.inputSize       = b.inputSize;
        this.hiddenSize      = b.hiddenSize;
        this.numLayers       = b.numLayers;
        this.forecastHorizon = b.forecastHorizon;
        this.epochs          = b.epochs;
        this.learningRate    = b.learningRate;
        this.dropout         = b.dropout;
        this.lookback        = b.lookback;
        this.gradientClip    = b.gradientClip;
        this.randomSeed      = b.randomSeed;
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public String toString() {
        return String.format(
            "LstmConfig[input=%d, hidden=%d, layers=%d, horizon=%d, lookback=%d, epochs=%d, lr=%.4f]",
            inputSize, hiddenSize, numLayers, forecastHorizon, lookback, epochs, learningRate);
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    public static final class Builder {
        private int    inputSize       = 1;
        private int    hiddenSize      = 64;
        private int    numLayers       = 1;
        private int    forecastHorizon = 1;
        private int    epochs          = 50;
        private double learningRate    = 1e-3;
        private double dropout         = 0.0;
        private int    lookback        = 24;
        private double gradientClip    = 5.0;
        private long   randomSeed      = 42L;

        public Builder inputSize(int v)       { this.inputSize = v;       return this; }
        public Builder hiddenSize(int v)      { this.hiddenSize = v;      return this; }
        public Builder numLayers(int v)       { this.numLayers = v;       return this; }
        public Builder forecastHorizon(int v) { this.forecastHorizon = v; return this; }
        public Builder epochs(int v)          { this.epochs = v;          return this; }
        public Builder learningRate(double v) { this.learningRate = v;    return this; }
        public Builder dropout(double v)      { this.dropout = v;         return this; }
        public Builder lookback(int v)        { this.lookback = v;        return this; }
        public Builder gradientClip(double v) { this.gradientClip = v;   return this; }
        public Builder randomSeed(long v)     { this.randomSeed = v;      return this; }

        public LstmConfig build() {
            if (inputSize <= 0)       throw new IllegalArgumentException("inputSize must be > 0");
            if (hiddenSize <= 0)      throw new IllegalArgumentException("hiddenSize must be > 0");
            if (numLayers <= 0)       throw new IllegalArgumentException("numLayers must be > 0");
            if (forecastHorizon <= 0) throw new IllegalArgumentException("forecastHorizon must be > 0");
            if (epochs <= 0)          throw new IllegalArgumentException("epochs must be > 0");
            if (learningRate <= 0)    throw new IllegalArgumentException("learningRate must be > 0");
            if (lookback <= 0)        throw new IllegalArgumentException("lookback must be > 0");
            return new LstmConfig(this);
        }
    }
}
