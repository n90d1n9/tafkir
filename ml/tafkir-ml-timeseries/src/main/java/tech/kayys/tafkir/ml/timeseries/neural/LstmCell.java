package tech.kayys.tafkir.ml.timeseries.neural;

import java.util.Random;

/**
 * Single-layer LSTM cell — pure Java, no tensor dependencies.
 *
 * <p>Implements the standard four-gate LSTM:
 * <ul>
 *   <li>f_t = σ(Wf·[h_{t−1}, x_t] + bf)  – forget gate</li>
 *   <li>i_t = σ(Wi·[h_{t−1}, x_t] + bi)  – input gate</li>
 *   <li>g_t = tanh(Wg·[h_{t−1}, x_t] + bg)  – candidate cell</li>
 *   <li>o_t = σ(Wo·[h_{t−1}, x_t] + bo)  – output gate</li>
 *   <li>c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t</li>
 *   <li>h_t = o_t ⊙ tanh(c_t)</li>
 * </ul>
 *
 * <p>Weights are stored as flattened row-major matrices:
 * <pre>
 *   W* shape: [hiddenSize × (hiddenSize + inputSize)]
 *   b* shape: [hiddenSize]
 * </pre>
 *
 * <p>Gradients are accumulated in the {@code d*} fields and applied
 * via {@link #applyGradients(double, double)}.
 */
public final class LstmCell {

    public final int inputSize;
    public final int hiddenSize;
    final int combined; // hiddenSize + inputSize

    // Weights [hiddenSize × combined], biases [hiddenSize]
    double[] Wf, Wi, Wg, Wo;
    double[] bf, bi, bg, bo;

    // Gradient accumulators
    double[] dWf, dWi, dWg, dWo;
    double[] dbf, dbi, dbg, dbo;

    // Adam moment estimates
    private double[] mWf, mWi, mWg, mWo, mbf, mbi, mbg, mbo;
    private double[] vWf, vWi, vWg, vWo, vbf, vbi, vbg, vbo;
    private int adamStep = 0;
    private static final double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;

    // Forward-pass cache (for BPTT)
    double[] f, i, g, o, c, h, cPrev, hPrev;
    double[] x; // concatenated [h_{t-1}, x_t]

    public LstmCell(int inputSize, int hiddenSize, long seed) {
        this.inputSize  = inputSize;
        this.hiddenSize = hiddenSize;
        this.combined   = hiddenSize + inputSize;
        initWeights(seed);
        initAdam();
    }

    // ── Weight initialisation ─────────────────────────────────────────────────

    private void initWeights(long seed) {
        Random rng = new Random(seed);
        double scale = Math.sqrt(2.0 / combined); // He init
        int sz = hiddenSize * combined;
        Wf = randArray(rng, sz, scale); Wi = randArray(rng, sz, scale);
        Wg = randArray(rng, sz, scale); Wo = randArray(rng, sz, scale);
        bf = new double[hiddenSize];    bi = new double[hiddenSize];
        bg = new double[hiddenSize];    bo = new double[hiddenSize];
        // Forget gate bias = 1 (helps with vanishing gradient)
        java.util.Arrays.fill(bf, 1.0);
        resetGradients();
    }

    private static double[] randArray(Random rng, int size, double scale) {
        double[] a = new double[size];
        for (int k = 0; k < size; k++) a[k] = rng.nextGaussian() * scale;
        return a;
    }

    private void initAdam() {
        int sz = hiddenSize * combined;
        mWf = new double[sz]; mWi = new double[sz]; mWg = new double[sz]; mWo = new double[sz];
        vWf = new double[sz]; vWi = new double[sz]; vWg = new double[sz]; vWo = new double[sz];
        mbf = new double[hiddenSize]; mbi = new double[hiddenSize];
        mbg = new double[hiddenSize]; mbo = new double[hiddenSize];
        vbf = new double[hiddenSize]; vbi = new double[hiddenSize];
        vbg = new double[hiddenSize]; vbo = new double[hiddenSize];
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * Single-step forward pass.
     *
     * @param xt     Input at time t  [inputSize]
     * @param hPrev  Previous hidden state [hiddenSize]
     * @param cPrev  Previous cell state [hiddenSize]
     * @return New hidden state h_t [hiddenSize]
     */
    public double[] forward(double[] xt, double[] hPrev, double[] cPrev) {
        this.hPrev = hPrev.clone();
        this.cPrev = cPrev.clone();

        // Concatenate [h_{t-1}, x_t]
        x = new double[combined];
        System.arraycopy(hPrev, 0, x, 0,           hiddenSize);
        System.arraycopy(xt,    0, x, hiddenSize,   inputSize);

        f = sigmoid(addBias(matMul(Wf, x), bf));
        i = sigmoid(addBias(matMul(Wi, x), bi));
        g = tanh(addBias(matMul(Wg, x), bg));
        o = sigmoid(addBias(matMul(Wo, x), bo));

        c = new double[hiddenSize];
        h = new double[hiddenSize];
        for (int k = 0; k < hiddenSize; k++) {
            c[k] = f[k] * cPrev[k] + i[k] * g[k];
            h[k] = o[k] * Math.tanh(c[k]);
        }
        return h;
    }

    /** Returns current cell state (call after forward). */
    public double[] getCellState() { return c.clone(); }

    // ── Backward pass (BPTT one step) ────────────────────────────────────────

    /**
     * Backpropagation through one time step.
     *
     * @param dhNext  Gradient of loss w.r.t. h_t (from dense layer + next cell step)
     * @param dcNext  Gradient of loss w.r.t. c_t (from next cell step)
     * @return {dh_prev, dc_prev} gradients to pass to the previous time step
     */
    public double[][] backward(double[] dhNext, double[] dcNext) {
        double[] tanhC = new double[hiddenSize];
        for (int k = 0; k < hiddenSize; k++) tanhC[k] = Math.tanh(c[k]);

        double[] do_ = new double[hiddenSize]; // output gate
        double[] dc  = new double[hiddenSize]; // cell
        double[] df  = new double[hiddenSize]; // forget gate
        double[] di  = new double[hiddenSize]; // input gate
        double[] dg  = new double[hiddenSize]; // candidate

        for (int k = 0; k < hiddenSize; k++) {
            do_[k] = dhNext[k] * tanhC[k];
            dc [k] = dhNext[k] * o[k] * (1 - tanhC[k] * tanhC[k]) + dcNext[k];
            df [k] = dc[k]  * cPrev[k];
            di [k] = dc[k]  * g[k];
            dg [k] = dc[k]  * i[k];
        }

        // Apply gate derivative
        double[] dof = gateGrad(do_, o, false); // sigmoid
        double[] dff = gateGrad(df,  f, false);
        double[] dif = gateGrad(di,  i, false);
        double[] dgf = gateGrad(dg,  g, true);  // tanh

        // Accumulate weight gradients
        outerAdd(dWf, dff, x); outerAdd(dWi, dif, x);
        outerAdd(dWg, dgf, x); outerAdd(dWo, dof, x);
        addTo(dbf, dff); addTo(dbi, dif); addTo(dbg, dgf); addTo(dbo, dof);

        // Propagate to previous step
        double[] dx = new double[combined];
        matMulTAdd(Wf, dff, dx); matMulTAdd(Wi, dif, dx);
        matMulTAdd(Wg, dgf, dx); matMulTAdd(Wo, dof, dx);

        double[] dhPrev = new double[hiddenSize];
        double[] dcPrev = new double[hiddenSize];
        System.arraycopy(dx, 0, dhPrev, 0, hiddenSize);
        for (int k = 0; k < hiddenSize; k++) dcPrev[k] = dc[k] * f[k];
        return new double[][]{dhPrev, dcPrev};
    }

    /** Element-wise gate gradient: sigmoid′(z) = g(1-g); tanh′(z) = 1-g². */
    private double[] gateGrad(double[] dOut, double[] gate, boolean isTanh) {
        double[] result = new double[hiddenSize];
        for (int k = 0; k < hiddenSize; k++) {
            result[k] = isTanh
                ? dOut[k] * (1 - gate[k] * gate[k])
                : dOut[k] * gate[k] * (1 - gate[k]);
        }
        return result;
    }

    // ── Gradient update ───────────────────────────────────────────────────────

    /** Apply Adam updates and reset gradient accumulators. */
    public void applyGradients(double lr, double maxNorm) {
        adamStep++;
        if (maxNorm > 0) clipGrads(maxNorm);
        adam(Wf, dWf, mWf, vWf, lr); adam(Wi, dWi, mWi, vWi, lr);
        adam(Wg, dWg, mWg, vWg, lr); adam(Wo, dWo, mWo, vWo, lr);
        adam(bf, dbf, mbf, vbf, lr); adam(bi, dbi, mbi, vbi, lr);
        adam(bg, dbg, mbg, vbg, lr); adam(bo, dbo, mbo, vbo, lr);
        resetGradients();
    }

    private void adam(double[] W, double[] dW, double[] m, double[] v, double lr) {
        double bc1 = 1 - Math.pow(BETA1, adamStep);
        double bc2 = 1 - Math.pow(BETA2, adamStep);
        for (int k = 0; k < W.length; k++) {
            m[k] = BETA1 * m[k] + (1 - BETA1) * dW[k];
            v[k] = BETA2 * v[k] + (1 - BETA2) * dW[k] * dW[k];
            W[k] -= lr * (m[k] / bc1) / (Math.sqrt(v[k] / bc2) + EPS);
        }
    }

    private void clipGrads(double maxNorm) {
        double norm = 0;
        for (double[] g : new double[][]{dWf, dWi, dWg, dWo, dbf, dbi, dbg, dbo})
            for (double v : g) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            for (double[] g : new double[][]{dWf, dWi, dWg, dWo, dbf, dbi, dbg, dbo})
                for (int k = 0; k < g.length; k++) g[k] *= scale;
        }
    }

    public void resetGradients() {
        int sz = hiddenSize * combined;
        dWf = new double[sz]; dWi = new double[sz]; dWg = new double[sz]; dWo = new double[sz];
        dbf = new double[hiddenSize]; dbi = new double[hiddenSize];
        dbg = new double[hiddenSize]; dbo = new double[hiddenSize];
    }

    // ── Math helpers ──────────────────────────────────────────────────────────

    /** Row-major matrix-vector product: W [rows×cols] × v [cols] → [rows] */
    double[] matMul(double[] W, double[] v) {
        double[] result = new double[hiddenSize];
        for (int row = 0; row < hiddenSize; row++) {
            double sum = 0;
            int off = row * combined;
            for (int col = 0; col < combined; col++) sum += W[off + col] * v[col];
            result[row] = sum;
        }
        return result;
    }

    /** Transpose matrix-vector product Wᵀ × v, accumulated into dst. */
    void matMulTAdd(double[] W, double[] v, double[] dst) {
        for (int row = 0; row < hiddenSize; row++) {
            int off = row * combined;
            for (int col = 0; col < combined; col++) dst[col] += W[off + col] * v[row];
        }
    }

    /** Outer product accumulate: dW += a ⊗ b */
    void outerAdd(double[] dW, double[] a, double[] b) {
        for (int row = 0; row < hiddenSize; row++) {
            int off = row * combined;
            for (int col = 0; col < combined; col++) dW[off + col] += a[row] * b[col];
        }
    }

    private static void addTo(double[] dst, double[] src) {
        for (int k = 0; k < dst.length; k++) dst[k] += src[k];
    }

    private static double[] addBias(double[] a, double[] b) {
        double[] r = a.clone();
        for (int k = 0; k < r.length; k++) r[k] += b[k];
        return r;
    }

    private static double[] sigmoid(double[] a) {
        double[] r = new double[a.length];
        for (int k = 0; k < a.length; k++) r[k] = 1.0 / (1.0 + Math.exp(-a[k]));
        return r;
    }

    private static double[] tanh(double[] a) {
        double[] r = new double[a.length];
        for (int k = 0; k < a.length; k++) r[k] = Math.tanh(a[k]);
        return r;
    }
}
