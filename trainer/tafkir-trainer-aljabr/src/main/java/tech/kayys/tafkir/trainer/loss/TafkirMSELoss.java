package tech.kayys.tafkir.trainer.loss;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

/**
 * Mean Squared Error loss.
 */
public final class TafkirMSELoss implements TafkirLoss {
    @Override
    public TafkirTensor compute(TafkirTensor pred, TafkirTensor target) {
        TafkirTensor diff = pred.sub(target);
        return diff.mul(diff).mean();
    }
}
