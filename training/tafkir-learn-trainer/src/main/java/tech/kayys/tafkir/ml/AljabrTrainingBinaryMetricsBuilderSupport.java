package tech.kayys.tafkir.ml;

/**
 * Fluent binary classification metric registration for training option builders.
 */
public abstract class AljabrTrainingBinaryMetricsBuilderSupport<
        B extends AljabrTrainingBinaryMetricsBuilderSupport<B>>
        extends AljabrTrainingClassificationMetricsBuilderSupport<B> {

    public B binaryAccuracyMetric() {
        return metric(Aljabr.DL.binaryAccuracyMetric());
    }

    public B binaryAccuracyMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryAccuracyMetric(logitThreshold));
    }

    public B binaryBalancedAccuracyMetric() {
        return metric(Aljabr.DL.binaryBalancedAccuracyMetric());
    }

    public B binaryBalancedAccuracyMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryBalancedAccuracyMetric(logitThreshold));
    }

    public B binaryMatthewsCorrelationCoefficientMetric() {
        return metric(Aljabr.DL.binaryMatthewsCorrelationCoefficientMetric());
    }

    public B binaryMatthewsCorrelationCoefficientMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryMatthewsCorrelationCoefficientMetric(logitThreshold));
    }

    public B binaryMccMetric() {
        return metric(Aljabr.DL.binaryMccMetric());
    }

    public B binaryMccMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryMccMetric(logitThreshold));
    }

    public B binaryCohensKappaMetric() {
        return metric(Aljabr.DL.binaryCohensKappaMetric());
    }

    public B binaryCohensKappaMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryCohensKappaMetric(logitThreshold));
    }

    public B binaryKappaMetric() {
        return metric(Aljabr.DL.binaryKappaMetric());
    }

    public B binaryKappaMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryKappaMetric(logitThreshold));
    }

    public B binaryConfusionMatrixMetric() {
        return metric(Aljabr.DL.binaryConfusionMatrixMetric());
    }

    public B binaryConfusionMatrixMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryConfusionMatrixMetric(logitThreshold));
    }

    public B binaryPrecisionMetric() {
        return metric(Aljabr.DL.binaryPrecisionMetric());
    }

    public B binaryPrecisionMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryPrecisionMetric(logitThreshold));
    }

    public B binaryRecallMetric() {
        return metric(Aljabr.DL.binaryRecallMetric());
    }

    public B binaryRecallMetric(float logitThreshold) {
        return metric(Aljabr.DL.binaryRecallMetric(logitThreshold));
    }

    public B binaryF1Metric() {
        return metric(Aljabr.DL.binaryF1Metric());
    }

    public B binaryF1Metric(float logitThreshold) {
        return metric(Aljabr.DL.binaryF1Metric(logitThreshold));
    }

    public B binaryRocAucMetric() {
        return metric(Aljabr.DL.binaryRocAucMetric());
    }

    public B binaryAurocMetric() {
        return metric(Aljabr.DL.binaryAurocMetric());
    }

    public B binaryAveragePrecisionMetric() {
        return metric(Aljabr.DL.binaryAveragePrecisionMetric());
    }

    public B binaryBestF1Metric() {
        return metric(Aljabr.DL.binaryBestF1Metric());
    }

    public B binaryBestF1ThresholdMetric() {
        return metric(Aljabr.DL.binaryBestF1ThresholdMetric());
    }

    public B binaryPrecisionAtRecallMetric(double minimumRecall) {
        return metric(Aljabr.DL.binaryPrecisionAtRecallMetric(minimumRecall));
    }

    public B binaryRecallAtPrecisionMetric(double minimumPrecision) {
        return metric(Aljabr.DL.binaryRecallAtPrecisionMetric(minimumPrecision));
    }

    public B binaryBrierScoreMetric() {
        return metric(Aljabr.DL.binaryBrierScoreMetric());
    }

    public B binaryLogLossMetric() {
        return metric(Aljabr.DL.binaryLogLossMetric());
    }

    public B binaryCrossEntropyMetric() {
        return metric(Aljabr.DL.binaryCrossEntropyMetric());
    }

    public B binaryExpectedCalibrationErrorMetric() {
        return metric(Aljabr.DL.binaryExpectedCalibrationErrorMetric());
    }

    public B binaryExpectedCalibrationErrorMetric(int bins) {
        return metric(Aljabr.DL.binaryExpectedCalibrationErrorMetric(bins));
    }

    public B binaryRankingMetrics() {
        return binaryRocAucMetric()
                .binaryAveragePrecisionMetric();
    }

    public B binaryImbalanceMetrics() {
        return binaryBalancedAccuracyMetric()
                .binaryMccMetric();
    }

    public B binaryImbalanceMetrics(float logitThreshold) {
        return binaryBalancedAccuracyMetric(logitThreshold)
                .binaryMccMetric(logitThreshold);
    }

    public B binaryAgreementMetrics() {
        return binaryKappaMetric();
    }

    public B binaryAgreementMetrics(float logitThreshold) {
        return binaryKappaMetric(logitThreshold);
    }

    public B binaryThresholdTuningMetrics() {
        return binaryBestF1ThresholdMetric();
    }

    public B binaryThresholdTuningMetrics(double minimumRecall, double minimumPrecision) {
        return binaryBestF1ThresholdMetric()
                .binaryPrecisionAtRecallMetric(minimumRecall)
                .binaryRecallAtPrecisionMetric(minimumPrecision);
    }

    public B binaryCalibrationMetrics() {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric();
    }

    public B binaryCalibrationMetrics(int bins) {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric(bins);
    }

    public B binaryClassificationMetrics() {
        return binaryAccuracyMetric()
                .binaryPrecisionMetric()
                .binaryRecallMetric()
                .binaryF1Metric();
    }

    public B binaryClassificationMetrics(float logitThreshold) {
        return binaryAccuracyMetric(logitThreshold)
                .binaryPrecisionMetric(logitThreshold)
                .binaryRecallMetric(logitThreshold)
                .binaryF1Metric(logitThreshold);
    }

    public B binaryMetrics() {
        return binaryClassificationMetrics();
    }

    public B binaryMetrics(float logitThreshold) {
        return binaryClassificationMetrics(logitThreshold);
    }
}
