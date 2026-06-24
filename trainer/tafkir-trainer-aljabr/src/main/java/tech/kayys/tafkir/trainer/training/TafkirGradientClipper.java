package tech.kayys.tafkir.trainer.training;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.List;

/**
 * Gradient clipping utility for Tafkir tensors.
 *
 * <p>Prevents exploding gradients by scaling down gradients if their global L2 norm
 * exceeds a specified threshold.
 */
public final class TafkirGradientClipper {

    private final double maxNorm;
    private double lastGlobalNorm;
    private double lastScale;

    /**
     * Creates a new gradient clipper.
     *
     * @param maxNorm maximum allowed global L2 norm
     */
    public TafkirGradientClipper(double maxNorm) {
        this.maxNorm = maxNorm;
        this.lastGlobalNorm = 0.0;
        this.lastScale = 1.0;
    }

    /**
     * Clips the gradients of the given parameters in-place if their global norm exceeds maxNorm.
     *
     * @param parameters list of tensors to clip
     * @return true if clipping occurred, false otherwise
     */
    public boolean clip(List<TafkirTensor> parameters) {
        double globalNormSq = 0.0;
        int gradCount = 0;

        // 1. Calculate global L2 norm squared
        for (TafkirTensor param : parameters) {
            if (!param.requiresGrad() || param.grad() == null) continue;
            
            TafkirTensor grad = param.grad();
            TafkirTensor squared = grad.mul(grad);
            float sumSq = squared.sum().toFloat();
            globalNormSq += sumSq;
            gradCount++;
        }

        if (gradCount == 0) return false;

        // 2. Compute true norm and scale
        double globalNorm = Math.sqrt(globalNormSq);
        this.lastGlobalNorm = globalNorm;

        if (globalNorm <= maxNorm || globalNorm == 0.0) {
            this.lastScale = 1.0;
            return false;
        }

        double scale = maxNorm / globalNorm;
        this.lastScale = scale;

        // 3. Apply scaling in-place
        for (TafkirTensor param : parameters) {
            if (!param.requiresGrad() || param.grad() == null) continue;
            
            TafkirTensor grad = param.grad();
            TafkirTensor scaledGrad = grad.mul((float) scale);
            param.setGrad(scaledGrad);
        }

        return true;
    }

    /** Returns the global norm computed during the last clip() call. */
    public double getLastGlobalNorm() { return lastGlobalNorm; }

    /** Returns the scaling factor applied during the last clip() call. */
    public double getLastScale() { return lastScale; }

    /** Returns the configured maximum norm. */
    public double getMaxNorm() { return maxNorm; }
}
