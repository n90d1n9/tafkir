package tech.kayys.tafkir.ml;

/**
 * Fluent regression metric registration for training option builders.
 */
public abstract class AljabrTrainingRegressionMetricsBuilderSupport<
        B extends AljabrTrainingRegressionMetricsBuilderSupport<B>>
        extends AljabrTrainingMultiLabelMetricsBuilderSupport<B> {

    public B meanAbsoluteErrorMetric() {
        return metric(Aljabr.DL.meanAbsoluteErrorMetric());
    }

    public B maeMetric() {
        return meanAbsoluteErrorMetric();
    }

    public B meanSquaredErrorMetric() {
        return metric(Aljabr.DL.meanSquaredErrorMetric());
    }

    public B mseMetric() {
        return meanSquaredErrorMetric();
    }

    public B rootMeanSquaredErrorMetric() {
        return metric(Aljabr.DL.rootMeanSquaredErrorMetric());
    }

    public B rmseMetric() {
        return rootMeanSquaredErrorMetric();
    }

    public B meanSquaredLogErrorMetric() {
        return metric(Aljabr.DL.meanSquaredLogErrorMetric());
    }

    public B msleMetric() {
        return meanSquaredLogErrorMetric();
    }

    public B rootMeanSquaredLogErrorMetric() {
        return metric(Aljabr.DL.rootMeanSquaredLogErrorMetric());
    }

    public B rmsleMetric() {
        return rootMeanSquaredLogErrorMetric();
    }

    public B meanPoissonDevianceMetric() {
        return metric(Aljabr.DL.meanPoissonDevianceMetric());
    }

    public B meanPoissonDevianceMetric(boolean logInput) {
        return metric(Aljabr.DL.meanPoissonDevianceMetric(logInput));
    }

    public B meanPoissonDevianceMetric(boolean logInput, double eps) {
        return metric(Aljabr.DL.meanPoissonDevianceMetric(logInput, eps));
    }

    public B poissonDevianceMetric() {
        return meanPoissonDevianceMetric();
    }

    public B poissonLogRateDevianceMetric() {
        return metric(Aljabr.DL.poissonLogRateDevianceMetric());
    }

    public B meanTweedieDevianceMetric() {
        return metric(Aljabr.DL.meanTweedieDevianceMetric());
    }

    public B meanTweedieDevianceMetric(double power) {
        return metric(Aljabr.DL.meanTweedieDevianceMetric(power));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput) {
        return metric(Aljabr.DL.meanTweedieDevianceMetric(power, logInput));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput, double eps) {
        return metric(Aljabr.DL.meanTweedieDevianceMetric(power, logInput, eps));
    }

    public B tweedieDevianceMetric() {
        return meanTweedieDevianceMetric();
    }

    public B compoundPoissonGammaDevianceMetric() {
        return metric(Aljabr.DL.compoundPoissonGammaDevianceMetric());
    }

    public B medianAbsoluteErrorMetric() {
        return metric(Aljabr.DL.medianAbsoluteErrorMetric());
    }

    public B medaeMetric() {
        return medianAbsoluteErrorMetric();
    }

    public B maxErrorMetric() {
        return metric(Aljabr.DL.maxErrorMetric());
    }

    public B pinballLossMetric(double quantile) {
        return metric(Aljabr.DL.pinballLossMetric(quantile));
    }

    public B meanPinballLossMetric(double quantile) {
        return metric(Aljabr.DL.meanPinballLossMetric(quantile));
    }

    public B predictionIntervalCoverageMetric() {
        return metric(Aljabr.DL.predictionIntervalCoverageMetric());
    }

    public B picpMetric() {
        return predictionIntervalCoverageMetric();
    }

    public B predictionIntervalMeanWidthMetric() {
        return metric(Aljabr.DL.predictionIntervalMeanWidthMetric());
    }

    public B predictionIntervalNormalizedMeanWidthMetric() {
        return metric(Aljabr.DL.predictionIntervalNormalizedMeanWidthMetric());
    }

    public B r2ScoreMetric() {
        return metric(Aljabr.DL.r2ScoreMetric());
    }

    public B r2Metric() {
        return r2ScoreMetric();
    }

    public B meanAbsolutePercentageErrorMetric() {
        return metric(Aljabr.DL.meanAbsolutePercentageErrorMetric());
    }

    public B mapeMetric() {
        return meanAbsolutePercentageErrorMetric();
    }

    public B symmetricMeanAbsolutePercentageErrorMetric() {
        return metric(Aljabr.DL.symmetricMeanAbsolutePercentageErrorMetric());
    }

    public B smapeMetric() {
        return symmetricMeanAbsolutePercentageErrorMetric();
    }

    public B meanBiasErrorMetric() {
        return metric(Aljabr.DL.meanBiasErrorMetric());
    }

    public B mbeMetric() {
        return meanBiasErrorMetric();
    }

    public B explainedVarianceMetric() {
        return metric(Aljabr.DL.explainedVarianceMetric());
    }

    public B explainedVarianceScoreMetric() {
        return explainedVarianceMetric();
    }

    public B regressionMetrics() {
        return meanAbsoluteErrorMetric()
                .meanSquaredErrorMetric()
                .rootMeanSquaredErrorMetric()
                .r2ScoreMetric();
    }

    public B regressionExtendedMetrics() {
        return regressionMetrics()
                .medianAbsoluteErrorMetric()
                .maxErrorMetric()
                .meanAbsolutePercentageErrorMetric()
                .symmetricMeanAbsolutePercentageErrorMetric()
                .meanBiasErrorMetric()
                .explainedVarianceMetric();
    }

    public B regressionLogScaleMetrics() {
        return meanSquaredLogErrorMetric()
                .rootMeanSquaredLogErrorMetric();
    }

    public B countRegressionMetrics() {
        return poissonLogRateDevianceMetric()
                .compoundPoissonGammaDevianceMetric();
    }

    public B regressionQuantileMetrics() {
        return pinballLossMetric(0.1)
                .pinballLossMetric(0.5)
                .pinballLossMetric(0.9);
    }

    public B regressionPredictionIntervalMetrics() {
        return predictionIntervalCoverageMetric()
                .predictionIntervalMeanWidthMetric()
                .predictionIntervalNormalizedMeanWidthMetric();
    }
}
