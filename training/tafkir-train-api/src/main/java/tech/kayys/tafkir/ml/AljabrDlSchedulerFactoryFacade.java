package tech.kayys.tafkir.ml;

import tech.kayys.tafkir.ml.optim.CosineAnnealingWarmRestartsLR;
import tech.kayys.tafkir.ml.optim.CosineAnnealingLR;
import tech.kayys.tafkir.ml.optim.ExponentialLR;
import tech.kayys.tafkir.ml.optim.LRScheduler;
import tech.kayys.tafkir.ml.optim.OneCycleLR;
import tech.kayys.tafkir.ml.optim.Optimizer;
import tech.kayys.tafkir.ml.optim.ReduceLROnPlateau;
import tech.kayys.tafkir.ml.optim.SequentialLR;
import tech.kayys.tafkir.ml.optim.StepLR;
import tech.kayys.tafkir.ml.optim.WarmupCosineScheduler;

import java.util.List;

/**
 * Learning-rate scheduler construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlSchedulerFactoryFacade extends AljabrDlOptimizerFactoryFacade {
    protected AljabrDlSchedulerFactoryFacade() {
    }

    public static StepLR stepScheduler(
            Optimizer optimizer,
            int stepSize,
            float gamma) {
        return new StepLR(optimizer, stepSize, gamma);
    }

    public static CosineAnnealingLR cosineAnnealingScheduler(
            Optimizer optimizer,
            int tMax,
            float minLr) {
        return new CosineAnnealingLR(optimizer, tMax, minLr);
    }

    public static ExponentialLR exponentialScheduler(
            Optimizer optimizer,
            float gamma) {
        return new ExponentialLR(optimizer, gamma);
    }

    public static WarmupCosineScheduler warmupCosineScheduler(
            Optimizer optimizer,
            int warmupSteps,
            int totalSteps,
            float maxLr,
            float minLr) {
        return new WarmupCosineScheduler(optimizer, warmupSteps, totalSteps, maxLr, minLr);
    }

    public static CosineAnnealingWarmRestartsLR cosineAnnealingWarmRestartsScheduler(
            Optimizer optimizer,
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        return new CosineAnnealingWarmRestartsLR(optimizer, firstCycleSteps, cycleMultiplier, minLr);
    }

    public static OneCycleLR oneCycleScheduler(
            Optimizer optimizer,
            int totalSteps,
            float maxLr) {
        return new OneCycleLR(optimizer, totalSteps, maxLr);
    }

    public static OneCycleLR oneCycleScheduler(
            Optimizer optimizer,
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor,
            OneCycleLR.AnnealStrategy annealStrategy) {
        return new OneCycleLR(
                optimizer,
                totalSteps,
                maxLr,
                pctStart,
                divFactor,
                finalDivFactor,
                annealStrategy);
    }

    public static ReduceLROnPlateau reduceLrOnPlateauScheduler(
            Optimizer optimizer,
            ReduceLROnPlateau.Mode mode,
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        return new ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, cooldown, minLr);
    }

    public static SequentialLR sequentialScheduler(
            Optimizer optimizer,
            List<? extends LRScheduler> schedulers,
            int... milestones) {
        return new SequentialLR(optimizer, List.copyOf(schedulers), milestones);
    }
}
