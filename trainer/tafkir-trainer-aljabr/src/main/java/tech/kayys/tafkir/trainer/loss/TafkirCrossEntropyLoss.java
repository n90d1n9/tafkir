package tech.kayys.tafkir.trainer.loss;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

/**
 * Cross-entropy loss for classification.
 * Expects pred to be raw logits [batch, numClasses].
 */
public final class TafkirCrossEntropyLoss implements TafkirLoss {
    @Override
    public TafkirTensor compute(TafkirTensor pred, TafkirTensor target) {
        return pred.logSoftmax(1).crossEntropy(target);
    }
}
