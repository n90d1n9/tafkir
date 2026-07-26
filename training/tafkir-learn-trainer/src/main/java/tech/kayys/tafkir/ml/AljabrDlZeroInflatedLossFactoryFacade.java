package tech.kayys.tafkir.ml;

import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedNegativeBinomialNllLoss;
import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedPoissonNllLoss;

/**
 * Zero-inflated distribution loss helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlZeroInflatedLossFactoryFacade extends AljabrDlCountLossFactoryFacade {
    protected AljabrDlZeroInflatedLossFactoryFacade() {
    }

    public static ZeroInflatedPoissonNllLoss zeroInflatedPoissonNllLoss() {
        return new ZeroInflatedPoissonNllLoss();
    }

    public static ZeroInflatedPoissonNllLoss zeroInflatedPoissonNllLoss(boolean logRateInput) {
        return new ZeroInflatedPoissonNllLoss(logRateInput);
    }

    public static ZeroInflatedPoissonNllLoss zeroInflatedPoissonNllLoss(
            boolean logRateInput,
            boolean includeConstant) {
        return new ZeroInflatedPoissonNllLoss(logRateInput, includeConstant);
    }

    public static ZeroInflatedPoissonNllLoss zeroInflatedPoissonNllLoss(
            boolean logRateInput,
            boolean includeConstant,
            double eps) {
        return new ZeroInflatedPoissonNllLoss(logRateInput, includeConstant, eps);
    }

    public static ZeroInflatedPoissonNllLoss zipNllLoss() {
        return zeroInflatedPoissonNllLoss();
    }

    public static ZeroInflatedPoissonNllLoss excessZeroCountNllLoss() {
        return zeroInflatedPoissonNllLoss();
    }

    public static ZeroInflatedNegativeBinomialNllLoss zeroInflatedNegativeBinomialNllLoss() {
        return new ZeroInflatedNegativeBinomialNllLoss();
    }

    public static ZeroInflatedNegativeBinomialNllLoss zeroInflatedNegativeBinomialNllLoss(boolean logInput) {
        return new ZeroInflatedNegativeBinomialNllLoss(logInput);
    }

    public static ZeroInflatedNegativeBinomialNllLoss zeroInflatedNegativeBinomialNllLoss(
            boolean logInput,
            boolean includeConstant) {
        return new ZeroInflatedNegativeBinomialNllLoss(logInput, includeConstant);
    }

    public static ZeroInflatedNegativeBinomialNllLoss zeroInflatedNegativeBinomialNllLoss(
            boolean logInput,
            boolean includeConstant,
            double eps) {
        return new ZeroInflatedNegativeBinomialNllLoss(logInput, includeConstant, eps);
    }

    public static ZeroInflatedNegativeBinomialNllLoss zinbNllLoss() {
        return zeroInflatedNegativeBinomialNllLoss();
    }

    public static ZeroInflatedNegativeBinomialNllLoss excessZeroOverdispersedCountNllLoss() {
        return zeroInflatedNegativeBinomialNllLoss();
    }
}
