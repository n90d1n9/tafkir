package tech.kayys.tafkir.ml.bytelatent;

import tech.kayys.tafkir.trainer.api.TrainingSummary;

/**
 * Minimal runtime session for byte-latent trainer flows.
 */
public interface ByteLatentTrainerSession extends AutoCloseable {

    ByteLatentTrainerConfig config();

    int currentEpoch();

    int globalStep();

    TrainingSummary fit();

    TrainingSummary summary();

    boolean isStopped();

    void stop();

    @Override
    default void close() {
        stop();
    }
}
