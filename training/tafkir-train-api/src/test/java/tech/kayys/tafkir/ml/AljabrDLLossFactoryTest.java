package tech.kayys.tafkir.ml;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.tafkir.ml.nn.loss.BinaryFocalWithLogitsLoss;
import tech.kayys.tafkir.ml.nn.loss.CausalLanguageModelingLoss;
import tech.kayys.tafkir.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.tafkir.ml.nn.loss.FocalLoss;
import tech.kayys.tafkir.ml.nn.loss.GaussianNllLoss;
import tech.kayys.tafkir.ml.nn.loss.HuberLoss;
import tech.kayys.tafkir.ml.nn.loss.L1Loss;
import tech.kayys.tafkir.ml.nn.loss.MSELoss;
import tech.kayys.tafkir.ml.nn.loss.NegativeBinomialNllLoss;
import tech.kayys.tafkir.ml.nn.loss.PinballLoss;
import tech.kayys.tafkir.ml.nn.loss.PoissonNllLoss;
import tech.kayys.tafkir.ml.nn.loss.PredictionIntervalLoss;
import tech.kayys.tafkir.ml.nn.loss.SmoothL1Loss;
import tech.kayys.tafkir.ml.nn.loss.TweedieNllLoss;
import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedNegativeBinomialNllLoss;
import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedPoissonNllLoss;

class AljabrDLLossFactoryTest {

    @Test
    void exposesClassificationLossFactoriesOnAljabrFacade() {
        assertInstanceOf(CrossEntropyLoss.class, Aljabr.DL.crossEntropy());
        assertInstanceOf(CrossEntropyLoss.class, Aljabr.DL.crossEntropy(new float[] {1.0f, 2.0f}));
        assertInstanceOf(CausalLanguageModelingLoss.class, Aljabr.DL.causalLanguageModelingLoss());
        assertInstanceOf(CausalLanguageModelingLoss.class, Aljabr.DL.causalLanguageModelingLoss(-1.0f));
        assertInstanceOf(FocalLoss.class, Aljabr.DL.focalLoss());
        assertInstanceOf(FocalLoss.class, Aljabr.DL.focalLoss(2.0f));
        assertInstanceOf(FocalLoss.class, Aljabr.DL.focalLoss(2.0f, 0.25f));
        assertInstanceOf(FocalLoss.class, Aljabr.DL.focalLoss(2.0f, new float[] {1.0f, 2.0f}));
    }

    @Test
    void exposesRegressionLossFactoriesOnAljabrFacade() {
        assertInstanceOf(MSELoss.class, Aljabr.DL.mseLoss());
        assertInstanceOf(L1Loss.class, Aljabr.DL.l1Loss());
        assertInstanceOf(HuberLoss.class, Aljabr.DL.huberLoss());
        assertInstanceOf(HuberLoss.class, Aljabr.DL.huberLoss(1.5f));
        assertInstanceOf(SmoothL1Loss.class, Aljabr.DL.smoothL1Loss());
        assertInstanceOf(SmoothL1Loss.class, Aljabr.DL.smoothL1Loss(0.75f));
        assertInstanceOf(PinballLoss.class, Aljabr.DL.pinballLoss());
        assertInstanceOf(PinballLoss.class, Aljabr.DL.quantileLoss(0.9));
        assertInstanceOf(PinballLoss.class, Aljabr.DL.quantileLoss(0.1, 0.5, 0.9));
        assertInstanceOf(PinballLoss.class, Aljabr.DL.predictionIntervalLoss(0.1, 0.9));
        assertInstanceOf(PredictionIntervalLoss.class, Aljabr.DL.intervalScoreLoss());
        assertInstanceOf(PredictionIntervalLoss.class, Aljabr.DL.winklerLoss(0.2, 12.0));
        assertInstanceOf(GaussianNllLoss.class, Aljabr.DL.gaussianNllLoss());
        assertInstanceOf(GaussianNllLoss.class, Aljabr.DL.heteroscedasticGaussianLoss(true));
    }

    @Test
    void exposesCountAndZeroInflatedLossFactoriesOnAljabrFacade() {
        assertInstanceOf(PoissonNllLoss.class, Aljabr.DL.poissonNllLoss());
        assertInstanceOf(PoissonNllLoss.class, Aljabr.DL.poissonNllLoss(true, true, 1e-8));
        assertInstanceOf(PoissonNllLoss.class, Aljabr.DL.countNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Aljabr.DL.tweedieNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Aljabr.DL.compoundPoissonGammaLoss(1.5));
        assertInstanceOf(NegativeBinomialNllLoss.class, Aljabr.DL.negativeBinomialNllLoss());
        assertInstanceOf(NegativeBinomialNllLoss.class, Aljabr.DL.overdispersedCountNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Aljabr.DL.zeroInflatedPoissonNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Aljabr.DL.zipNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Aljabr.DL.excessZeroCountNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Aljabr.DL.zeroInflatedNegativeBinomialNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Aljabr.DL.zinbNllLoss());
        assertInstanceOf(
                ZeroInflatedNegativeBinomialNllLoss.class,
                Aljabr.DL.excessZeroOverdispersedCountNllLoss());
    }

    @Test
    void exposesBinaryLogitLossFactoriesOnAljabrFacade() {
        assertInstanceOf(BCEWithLogitsLoss.class, Aljabr.DL.bceWithLogitsLoss());
        assertInstanceOf(BCEWithLogitsLoss.class, Aljabr.DL.bceWithLogitsLoss(3.0f));
        assertInstanceOf(BCEWithLogitsLoss.class, Aljabr.DL.bceWithLogitsLoss(new float[] {2.0f, 3.0f}));
        assertInstanceOf(BCEWithLogitsLoss.class, Aljabr.DL.binaryCrossEntropyWithLogits());
        assertInstanceOf(BCEWithLogitsLoss.class, Aljabr.DL.binaryCrossEntropyWithLogits(2.0f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Aljabr.DL.binaryFocalWithLogitsLoss());
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Aljabr.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Aljabr.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, 3.0f));
        assertInstanceOf(
                BinaryFocalWithLogitsLoss.class,
                Aljabr.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, new float[] {2.0f, 3.0f}));
    }
}
