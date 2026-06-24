package tech.kayys.tafkir.trainer.training;

import java.util.function.Consumer;

/**
 * Early stopping utility for Tafkir training loops.
 *
 * <p>Monitors a metric (like validation loss) and signals to stop training
 * if the metric doesn't improve for {@code patience} consecutive epochs.
 */
public final class TafkirEarlyStopping {

    public enum Mode { MIN, MAX }

    private final int patience;
    private final double delta;
    private final Mode mode;
    private final Consumer<CheckpointMeta> checkpointCallback;

    private double bestScore;
    private int counter;
    private boolean stopped;
    private int bestEpoch;
    private CheckpointMeta bestCheckpoint;

    /**
     * Creates an early stopping monitor.
     *
     * @param patience epochs to wait for improvement before stopping
     * @param delta    minimum change to qualify as an improvement
     * @param mode     whether to minimize (e.g. loss) or maximize (e.g. accuracy)
     * @param callback optional callback invoked when early stopping triggers, providing the best checkpoint
     */
    public TafkirEarlyStopping(int patience, double delta, Mode mode, Consumer<CheckpointMeta> callback) {
        this.patience = patience;
        this.delta = delta;
        this.mode = mode;
        this.checkpointCallback = callback;
        this.bestScore = (mode == Mode.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        this.counter = 0;
        this.stopped = false;
        this.bestEpoch = -1;
    }

    /**
     * Creates a simple early stopping monitor aiming to minimize a metric (e.g. loss).
     *
     * @param patience epochs to wait for improvement
     */
    public TafkirEarlyStopping(int patience) {
        this(patience, 0.0, Mode.MIN, null);
    }

    /**
     * Checks if training should stop.
     *
     * @param score      current metric score
     * @param epoch      current epoch
     * @param checkpoint optional checkpoint metadata to save if this is the best epoch
     * @return true if training should stop, false otherwise
     */
    public boolean check(double score, int epoch, CheckpointMeta checkpoint) {
        if (stopped) return true;

        boolean improved = (mode == Mode.MIN)
                ? (score < bestScore - delta)
                : (score > bestScore + delta);

        if (improved) {
            bestScore = score;
            counter = 0;
            bestEpoch = epoch;
            if (checkpoint != null) {
                bestCheckpoint = checkpoint;
            }
        } else {
            counter++;
        }

        if (counter >= patience) {
            stopped = true;
            if (bestCheckpoint != null && checkpointCallback != null) {
                checkpointCallback.accept(bestCheckpoint);
            }
            return true;
        }

        return false;
    }

    public boolean isStopped() { return stopped; }
    public double getBestScore() { return bestScore; }
    public int getBestEpoch() { return bestEpoch; }
    public int getCounter() { return counter; }

    public void reset() {
        counter = 0;
        stopped = false;
        bestScore = (mode == Mode.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        bestEpoch = -1;
        bestCheckpoint = null;
    }

    /** Minimal record for callback reference. */
    public record CheckpointMeta(int epoch, double score, String path) {}
}
