package tech.kayys.tafkir.trainer.scheduler;

/**
 * Interface for learning rate schedulers that adjust a {@link
 * tech.kayys.tafkir.trainer.optim.TafkirOptimizer TafkirOptimizer}'s
 * learning rate during training.
 *
 * <h3>Epoch-level usage</h3>
 * <pre>{@code
 * TafkirLRScheduler scheduler = new TafkirCosineAnnealingLR(optimizer, epochs, 1e-6f);
 * for (int epoch = 0; epoch < epochs; epoch++) {
 *     scheduler.step(epoch);
 *     trainer.fit(x, y);
 * }
 * }</pre>
 *
 * <h3>Batch-level usage (warmup)</h3>
 * <pre>{@code
 * TafkirLRScheduler scheduler = new TafkirLinearWarmupCosineDecay(opt, 100, totalSteps, maxLR);
 * int globalStep = 0;
 * for (Batch batch : dataLoader) {
 *     scheduler.stepBatch(globalStep++);
 *     // ... forward / backward / step
 * }
 * }</pre>
 */
public interface TafkirLRScheduler {

    /**
     * Updates the learning rate based on the current epoch (0-indexed).
     *
     * @param epoch current training epoch
     */
    void step(int epoch);

    /**
     * Updates the learning rate based on the global batch step.
     * Override for per-step schedulers (e.g. linear warmup).
     *
     * @param globalStep the global batch step number (0-indexed)
     */
    default void stepBatch(int globalStep) {
        // No-op by default; override for per-step scheduling.
    }

    /**
     * Returns the learning rate currently applied to the optimizer.
     *
     * @return current learning rate
     */
    double getCurrentLR();

    /**
     * Resets the scheduler to its initial state.
     */
    void reset();

    /**
     * Returns a human-readable description for logging.
     *
     * @return description string
     */
    String getDescription();
}
