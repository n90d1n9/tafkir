package tech.kayys.tafkir.trainer.loss;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

/**
 * Cross-entropy loss for classification.
 *
 * <p>Expects pred to be raw logits [batch, numClasses].
 * Internally applies log_softmax + nll_loss.
 */
public final class TafkirCrossEntropyLoss implements TafkirLoss {
    @Override
    public TafkirTensor compute(TafkirTensor pred, TafkirTensor target) {
        return pred.logSoftmax(1).crossEntropy(target);
    }
}
