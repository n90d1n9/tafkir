package tech.kayys.tafkir.trainer.scheduler;

import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

/**
 * Step learning rate scheduler.
 *
 * <p>Decays the learning rate of the optimizer by {@code gamma} every {@code stepSize} epochs.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * TafkirLRScheduler scheduler = new TafkirStepLR(optimizer, 30, 0.1f);
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     scheduler.step(epoch);
 *     trainer.fit(x, y);
 * }
 * }</pre>
 */
public final class TafkirStepLR implements TafkirLRScheduler {

    private final TafkirOptimizer optimizer;
    private final double initialLR;
    private final int stepSize;
    private final double gamma;

    private double currentLR;

    /**
     * Creates a new step learning rate scheduler.
     *
     * @param optimizer the optimizer to control
     * @param stepSize  period of learning rate decay (in epochs)
     * @param gamma     multiplicative factor of learning rate decay
     */
    public TafkirStepLR(TafkirOptimizer optimizer, int stepSize, double gamma) {
        this.optimizer = optimizer;
        this.initialLR = optimizer.getLearningRate();
        this.stepSize  = stepSize;
        this.gamma     = gamma;
        this.currentLR = initialLR;
    }

    @Override
    public void step(int epoch) {
        if (epoch > 0 && epoch % stepSize == 0) {
            currentLR *= gamma;
            optimizer.setLearningRate((float) currentLR);
        }
    }

    @Override
    public double getCurrentLR() {
        return currentLR;
    }

    @Override
    public void reset() {
        currentLR = initialLR;
        optimizer.setLearningRate((float) initialLR);
    }

    @Override
    public String getDescription() {
        return String.format("StepLR(initial=%.6f, step_size=%d, gamma=%.2f)", initialLR, stepSize, gamma);
    }
}
