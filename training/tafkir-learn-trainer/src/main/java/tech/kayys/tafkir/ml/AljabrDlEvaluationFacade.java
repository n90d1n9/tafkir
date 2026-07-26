package tech.kayys.tafkir.ml;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import tech.kayys.tafkir.ml.Aljabr.DL.EvaluationOptions;
import tech.kayys.tafkir.ml.Aljabr.DL.EvaluationSummary;
import tech.kayys.tafkir.ml.Aljabr.DL.TrainingPreset;
import tech.kayys.tafkir.ml.nn.NNModule;
import tech.kayys.tafkir.ml.train.CanonicalTrainer;
import tech.kayys.tafkir.ml.train.TrainingLossFunction;
import tech.kayys.tafkir.train.data.DataLoader.Batch;

/**
 * Evaluation helpers exposed through {@link Aljabr.DL}.
 */
@SuppressWarnings("deprecation")
public class AljabrDlEvaluationFacade extends AljabrDlFactoryFacade {
    protected AljabrDlEvaluationFacade() {
    }

    public static EvaluationOptions.Builder evaluationOptions() {
        return EvaluationOptions.builder();
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, preset, evaluationOptions().build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            String deviceId,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, preset, evaluationOptions().device(deviceId).build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingPreset preset,
            EvaluationOptions options,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        Objects.requireNonNull(preset, "preset must not be null");
        return evaluate(model, loader, Aljabr.DL.presetLoss(preset), options, metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, loss, evaluationOptions().build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            String deviceId,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        return evaluate(model, loader, loss, evaluationOptions().device(deviceId).build(), metrics);
    }

    @SafeVarargs
    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            EvaluationOptions options,
            java.util.function.Supplier<? extends CanonicalTrainer.Metric>... metrics) {
        List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories =
                metrics == null ? List.of() : Arrays.asList(metrics);
        return evaluate(model, loader, loss, options, metricFactories);
    }

    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            String deviceId,
            List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
        return evaluate(model, loader, loss, evaluationOptions().device(deviceId).build(), metricFactories);
    }

    public static EvaluationSummary evaluate(
            NNModule model,
            Iterable<Batch> loader,
            TrainingLossFunction loss,
            EvaluationOptions options,
            List<java.util.function.Supplier<? extends CanonicalTrainer.Metric>> metricFactories) {
        return AljabrEvaluationRuntime.evaluate(model, loader, loss, options, metricFactories);
    }
}
