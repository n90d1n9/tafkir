package tech.kayys.tafkir.trainer.scheduler;

import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

/**
 * Cosine annealing learning rate scheduler with optional warm restarts (SGDR).
 *
 * <p>Learning rate follows:
 * <pre>
 *   lr = etaMin + (initialLR - etaMin) × (1 + cos(π × epoch / tMax)) / 2
 * </pre>
 *
 * <p>With warm restarts ({@code warmRestart=true}) the cosine cycle resets
 * every {@code restartPeriod} epochs, effectively implementing SGDR
 * (Loshchilov &amp; Hutter, 2017).
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var scheduler = new TafkirCosineAnnealingLR(optimizer, 100, 1e-6f);
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     scheduler.step(epoch);
 *     trainer.fit(x, y);
 * }
 * }</pre>
 */
public final class TafkirCosineAnnealingLR implements TafkirLRScheduler {

    private final TafkirOptimizer optimizer;
    private final double initialLR;
    private final double etaMin;
    private final int tMax;
    private final boolean warmRestart;
    private final int restartPeriod;

    private double currentLR;
    private int restartCount;

    /**
     * Creates a cosine annealing scheduler without warm restarts.
     *
     * @param optimizer the optimizer to control
     * @param tMax      total number of epochs in one cosine cycle
     * @param etaMin    minimum learning rate (floor)
     */
    public TafkirCosineAnnealingLR(TafkirOptimizer optimizer, int tMax, float etaMin) {
        this(optimizer, tMax, etaMin, false, tMax);
    }

    /**
     * Creates a cosine annealing scheduler with optional warm restarts.
     *
     * @param optimizer     the optimizer to control
     * @param tMax          total epochs per cosine period
     * @param etaMin        minimum learning rate
     * @param warmRestart   whether to enable periodic restarts
     * @param restartPeriod period between restarts (only used when {@code warmRestart=true})
     */
    public TafkirCosineAnnealingLR(TafkirOptimizer optimizer, int tMax, float etaMin,
                                    boolean warmRestart, int restartPeriod) {
        this.optimizer    = optimizer;
        this.initialLR    = optimizer.getLearningRate();
        this.etaMin       = etaMin;
        this.tMax         = tMax;
        this.warmRestart  = warmRestart;
        this.restartPeriod = restartPeriod;
        this.currentLR    = initialLR;
        this.restartCount = 0;
    }

    @Override
    public void step(int epoch) {
        int effective = warmRestart ? epoch % restartPeriod : epoch;
        restartCount  = warmRestart ? epoch / restartPeriod : 0;
        double progress = (double) effective / tMax;
        currentLR = etaMin + (initialLR - etaMin) * (1.0 + Math.cos(Math.PI * progress)) / 2.0;
        optimizer.setLearningRate((float) currentLR);
    }

    @Override
    public double getCurrentLR() { return currentLR; }

    @Override
    public void reset() {
        restartCount = 0;
        currentLR    = initialLR;
        optimizer.setLearningRate((float) initialLR);
    }

    @Override
    public String getDescription() {
        return String.format("CosineAnnealingLR(initial=%.6f, T_max=%d, eta_min=%.2e, warm_restart=%b)",
                initialLR, tMax, etaMin, warmRestart);
    }

    /** Returns the number of cosine restarts completed so far. */
    public int getRestartCount() { return restartCount; }
}
