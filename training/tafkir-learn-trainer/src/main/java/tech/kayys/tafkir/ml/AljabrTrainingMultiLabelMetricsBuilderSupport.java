package tech.kayys.tafkir.ml;

/**
 * Fluent multi-label metric registration for training option builders.
 */
public abstract class AljabrTrainingMultiLabelMetricsBuilderSupport<
        B extends AljabrTrainingMultiLabelMetricsBuilderSupport<B>>
        extends AljabrTrainingBinaryMetricsBuilderSupport<B> {

    public B multiLabelExactMatchMetric() {
        return metric(Aljabr.DL.multiLabelExactMatchMetric());
    }

    public B multiLabelExactMatchMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelExactMatchMetric(logitThreshold));
    }

    public B multiLabelHammingLossMetric() {
        return metric(Aljabr.DL.multiLabelHammingLossMetric());
    }

    public B multiLabelHammingLossMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelHammingLossMetric(logitThreshold));
    }

    public B multiLabelConfusionMatrixMetric() {
        return metric(Aljabr.DL.multiLabelConfusionMatrixMetric());
    }

    public B multiLabelConfusionMatrixMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelConfusionMatrixMetric(logitThreshold));
    }

    public B multiLabelMicroPrecisionMetric() {
        return metric(Aljabr.DL.multiLabelMicroPrecisionMetric());
    }

    public B multiLabelMicroPrecisionMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMicroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMicroRecallMetric() {
        return metric(Aljabr.DL.multiLabelMicroRecallMetric());
    }

    public B multiLabelMicroRecallMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMicroRecallMetric(logitThreshold));
    }

    public B multiLabelMicroF1Metric() {
        return metric(Aljabr.DL.multiLabelMicroF1Metric());
    }

    public B multiLabelMicroF1Metric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMicroF1Metric(logitThreshold));
    }

    public B multiLabelSamplePrecisionMetric() {
        return metric(Aljabr.DL.multiLabelSamplePrecisionMetric());
    }

    public B multiLabelSamplePrecisionMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelSamplePrecisionMetric(logitThreshold));
    }

    public B multiLabelSampleRecallMetric() {
        return metric(Aljabr.DL.multiLabelSampleRecallMetric());
    }

    public B multiLabelSampleRecallMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelSampleRecallMetric(logitThreshold));
    }

    public B multiLabelSampleF1Metric() {
        return metric(Aljabr.DL.multiLabelSampleF1Metric());
    }

    public B multiLabelSampleF1Metric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelSampleF1Metric(logitThreshold));
    }

    public B multiLabelSampleJaccardMetric() {
        return metric(Aljabr.DL.multiLabelSampleJaccardMetric());
    }

    public B multiLabelSampleJaccardMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelSampleJaccardMetric(logitThreshold));
    }

    public B multiLabelMacroPrecisionMetric() {
        return metric(Aljabr.DL.multiLabelMacroPrecisionMetric());
    }

    public B multiLabelMacroPrecisionMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMacroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMacroRecallMetric() {
        return metric(Aljabr.DL.multiLabelMacroRecallMetric());
    }

    public B multiLabelMacroRecallMetric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMacroRecallMetric(logitThreshold));
    }

    public B multiLabelMacroF1Metric() {
        return metric(Aljabr.DL.multiLabelMacroF1Metric());
    }

    public B multiLabelMacroF1Metric(float logitThreshold) {
        return metric(Aljabr.DL.multiLabelMacroF1Metric(logitThreshold));
    }

    public B multiLabelMacroRocAucMetric() {
        return metric(Aljabr.DL.multiLabelMacroRocAucMetric());
    }

    public B multiLabelMacroAurocMetric() {
        return metric(Aljabr.DL.multiLabelMacroAurocMetric());
    }

    public B multiLabelMacroAveragePrecisionMetric() {
        return metric(Aljabr.DL.multiLabelMacroAveragePrecisionMetric());
    }

    public B multiLabelLabelRankingAveragePrecisionMetric() {
        return metric(Aljabr.DL.multiLabelLabelRankingAveragePrecisionMetric());
    }

    public B multiLabelLrapMetric() {
        return metric(Aljabr.DL.multiLabelLrapMetric());
    }

    public B multiLabelRankingLossMetric() {
        return metric(Aljabr.DL.multiLabelRankingLossMetric());
    }

    public B multiLabelCoverageErrorMetric() {
        return metric(Aljabr.DL.multiLabelCoverageErrorMetric());
    }

    public B multiLabelMacroBestF1Metric() {
        return metric(Aljabr.DL.multiLabelMacroBestF1Metric());
    }

    public B multiLabelBestF1ThresholdsMetric() {
        return metric(Aljabr.DL.multiLabelBestF1ThresholdsMetric());
    }

    public B multiLabelRankingMetrics() {
        return multiLabelMacroRocAucMetric()
                .multiLabelMacroAveragePrecisionMetric()
                .multiLabelLabelRankingAveragePrecisionMetric()
                .multiLabelRankingLossMetric()
                .multiLabelCoverageErrorMetric();
    }

    public B multiLabelThresholdTuningMetrics() {
        return multiLabelBestF1ThresholdsMetric();
    }

    public B multiLabelMicroMetrics() {
        return multiLabelMicroPrecisionMetric()
                .multiLabelMicroRecallMetric()
                .multiLabelMicroF1Metric();
    }

    public B multiLabelMicroMetrics(float logitThreshold) {
        return multiLabelMicroPrecisionMetric(logitThreshold)
                .multiLabelMicroRecallMetric(logitThreshold)
                .multiLabelMicroF1Metric(logitThreshold);
    }

    public B multiLabelSampleMetrics() {
        return multiLabelSamplePrecisionMetric()
                .multiLabelSampleRecallMetric()
                .multiLabelSampleF1Metric()
                .multiLabelSampleJaccardMetric();
    }

    public B multiLabelSampleMetrics(float logitThreshold) {
        return multiLabelSamplePrecisionMetric(logitThreshold)
                .multiLabelSampleRecallMetric(logitThreshold)
                .multiLabelSampleF1Metric(logitThreshold)
                .multiLabelSampleJaccardMetric(logitThreshold);
    }

    public B multiLabelBinaryMetrics() {
        return multiLabelExactMatchMetric()
                .multiLabelHammingLossMetric()
                .multiLabelMicroMetrics()
                .multiLabelSampleMetrics()
                .multiLabelMacroPrecisionMetric()
                .multiLabelMacroRecallMetric()
                .multiLabelMacroF1Metric();
    }

    public B multiLabelBinaryMetrics(float logitThreshold) {
        return multiLabelExactMatchMetric(logitThreshold)
                .multiLabelHammingLossMetric(logitThreshold)
                .multiLabelMicroMetrics(logitThreshold)
                .multiLabelSampleMetrics(logitThreshold)
                .multiLabelMacroPrecisionMetric(logitThreshold)
                .multiLabelMacroRecallMetric(logitThreshold)
                .multiLabelMacroF1Metric(logitThreshold);
    }

    public B multiLabelMetrics() {
        return multiLabelBinaryMetrics();
    }

    public B multiLabelMetrics(float logitThreshold) {
        return multiLabelBinaryMetrics(logitThreshold);
    }
}
