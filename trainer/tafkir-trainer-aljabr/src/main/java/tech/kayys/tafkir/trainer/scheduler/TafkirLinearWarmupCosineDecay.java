package tech.kayys.tafkir.trainer.scheduler;

import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

/**
 * Learning rate scheduler that combines linear warmup with cosine decay.
 *
 * <p>The learning rate increases linearly from 0 to {@code maxLR} over {@code warmupSteps},
 * then decays following a cosine curve to {@code minLR} over the remaining {@code totalSteps}.
 * This scheduler should be called per batch using {@link #stepBatch(int)}.
 */
public final class TafkirLinearWarmupCosineDecay implements TafkirLRScheduler {

    private final TafkirOptimizer optimizer;
    private final int warmupSteps;
    private final int totalSteps;
    private final double maxLR;
    private final double minLR;

    private double currentLR;

    /**
     * Creates a linear warmup with cosine decay scheduler.
     *
     * @param optimizer   the optimizer to control
     * @param warmupSteps number of steps to linearly warmup the learning rate
     * @param totalSteps  total number of training steps (batches)
     * @param maxLR       maximum learning rate (at end of warmup)
     * @param minLR       minimum learning rate (at end of training)
     */
    public TafkirLinearWarmupCosineDecay(TafkirOptimizer optimizer, int warmupSteps, int totalSteps,
                                          double maxLR, double minLR) {
        this.optimizer   = optimizer;
        this.warmupSteps = warmupSteps;
        this.totalSteps  = totalSteps;
        this.maxLR       = maxLR;
        this.minLR       = minLR;
        this.currentLR   = 0.0;
        
        // Initial set if warmup steps > 0
        if (warmupSteps > 0) {
            optimizer.setLearningRate(0.0f);
        } else {
            this.currentLR = maxLR;
            optimizer.setLearningRate((float) maxLR);
        }
    }

    @Override
    public void step(int epoch) {
        // Typically not used per epoch, but we can no-op or fall back.
    }

    @Override
    public void stepBatch(int globalStep) {
        if (globalStep < warmupSteps) {
            // Linear warmup
            currentLR = maxLR * ((double) globalStep / warmupSteps);
        } else if (globalStep >= totalSteps) {
            currentLR = minLR;
        } else {
            // Cosine decay
            int decayStep = globalStep - warmupSteps;
            int decaySteps = totalSteps - warmupSteps;
            double progress = (double) decayStep / decaySteps;
            currentLR = minLR + 0.5 * (maxLR - minLR) * (1.0 + Math.cos(Math.PI * progress));
        }
        optimizer.setLearningRate((float) currentLR);
    }

    @Override
    public double getCurrentLR() {
        return currentLR;
    }

    @Override
    public void reset() {
        currentLR = warmupSteps > 0 ? 0.0 : maxLR;
        optimizer.setLearningRate((float) currentLR);
    }

    @Override
    public String getDescription() {
        return String.format("LinearWarmupCosineDecay(warmup=%d, total=%d, max_lr=%.6f)",
                warmupSteps, totalSteps, maxLR);
    }
}
