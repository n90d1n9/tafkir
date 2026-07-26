package tech.kayys.tafkir.ml.train;

import tech.kayys.tafkir.ml.autograd.GradTensor;

/**
 * Loss contract used by trainer implementations.
 */
@FunctionalInterface
public interface TrainingLossFunction {
    GradTensor compute(GradTensor predictions, GradTensor targets);
}
