package tech.kayys.tafkir.ml;

/**
 * Fluent multi-class classification metric registration for training option builders.
 */
public abstract class AljabrTrainingClassificationMetricsBuilderSupport<
        B extends AljabrTrainingClassificationMetricsBuilderSupport<B>>
        extends AljabrTrainingMetricRegistrySupport<B> {

    public B accuracyMetric() {
        return metric(Aljabr.DL.accuracyMetric());
    }

    public B confusionMatrixMetric() {
        return metric(Aljabr.DL.confusionMatrixMetric());
    }

    public B classificationConfusionMatrixMetric() {
        return metric(Aljabr.DL.classificationConfusionMatrixMetric());
    }

    public B topKAccuracyMetric(int k) {
        return metric(Aljabr.DL.topKAccuracyMetric(k));
    }

    public B classificationLogLossMetric() {
        return metric(Aljabr.DL.classificationLogLossMetric());
    }

    public B classificationCrossEntropyMetric() {
        return metric(Aljabr.DL.classificationCrossEntropyMetric());
    }

    public B classificationBalancedAccuracyMetric() {
        return metric(Aljabr.DL.classificationBalancedAccuracyMetric());
    }

    public B balancedAccuracyMetric() {
        return metric(Aljabr.DL.balancedAccuracyMetric());
    }

    public B classificationMatthewsCorrelationCoefficientMetric() {
        return metric(Aljabr.DL.classificationMatthewsCorrelationCoefficientMetric());
    }

    public B classificationMccMetric() {
        return metric(Aljabr.DL.classificationMccMetric());
    }

    public B matthewsCorrelationCoefficientMetric() {
        return metric(Aljabr.DL.matthewsCorrelationCoefficientMetric());
    }

    public B mccMetric() {
        return metric(Aljabr.DL.mccMetric());
    }

    public B classificationWeightedPrecisionMetric() {
        return metric(Aljabr.DL.classificationWeightedPrecisionMetric());
    }

    public B classificationWeightedRecallMetric() {
        return metric(Aljabr.DL.classificationWeightedRecallMetric());
    }

    public B classificationWeightedF1Metric() {
        return metric(Aljabr.DL.classificationWeightedF1Metric());
    }

    public B classificationCohensKappaMetric() {
        return metric(Aljabr.DL.classificationCohensKappaMetric());
    }

    public B classificationKappaMetric() {
        return metric(Aljabr.DL.classificationKappaMetric());
    }

    public B cohensKappaMetric() {
        return metric(Aljabr.DL.cohensKappaMetric());
    }

    public B kappaMetric() {
        return metric(Aljabr.DL.kappaMetric());
    }

    public B precisionMetric() {
        return metric(Aljabr.DL.precisionMetric());
    }

    public B recallMetric() {
        return metric(Aljabr.DL.recallMetric());
    }

    public B f1Metric() {
        return metric(Aljabr.DL.f1Metric());
    }

    public B macroF1Metric() {
        return metric(Aljabr.DL.macroF1Metric());
    }

    public B classificationMetrics() {
        return accuracyMetric()
                .classificationLogLossMetric()
                .precisionMetric()
                .recallMetric()
                .f1Metric();
    }

    public B classificationMacroRocAucMetric() {
        return metric(Aljabr.DL.classificationMacroRocAucMetric());
    }

    public B classificationMacroAurocMetric() {
        return metric(Aljabr.DL.classificationMacroAurocMetric());
    }

    public B classificationMacroAveragePrecisionMetric() {
        return metric(Aljabr.DL.classificationMacroAveragePrecisionMetric());
    }

    public B classificationRankingMetrics() {
        return classificationMacroRocAucMetric()
                .classificationMacroAveragePrecisionMetric();
    }

    public B classificationImbalanceMetrics() {
        return classificationBalancedAccuracyMetric()
                .classificationMccMetric()
                .classificationWeightedPrecisionMetric()
                .classificationWeightedRecallMetric()
                .classificationWeightedF1Metric();
    }

    public B classificationAgreementMetrics() {
        return classificationKappaMetric();
    }

    public B classificationBrierScoreMetric() {
        return metric(Aljabr.DL.classificationBrierScoreMetric());
    }

    public B classificationExpectedCalibrationErrorMetric() {
        return metric(Aljabr.DL.classificationExpectedCalibrationErrorMetric());
    }

    public B classificationExpectedCalibrationErrorMetric(int bins) {
        return metric(Aljabr.DL.classificationExpectedCalibrationErrorMetric(bins));
    }

    public B classificationCalibrationMetrics() {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric();
    }

    public B classificationCalibrationMetrics(int bins) {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric(bins);
    }
}
