package tech.kayys.tafkir.ml.metrics;

/**
 * Common ML evaluation metrics.
 */
public final class Metrics {

    private Metrics() {
    }

    public static double accuracy(int[] trueLabels, int[] predicted) {
        if (trueLabels.length != predicted.length) {
            throw new IllegalArgumentException("Array length mismatch");
        }
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == predicted[i]) correct++;
        }
        return (double) correct / trueLabels.length;
    }

    public static double precision(int[] trueLabels, int[] predicted, int positiveClass) {
        int tp = 0, fp = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (predicted[i] == positiveClass) {
                if (trueLabels[i] == positiveClass) tp++; else fp++;
            }
        }
        return (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
    }

    public static double recall(int[] trueLabels, int[] predicted, int positiveClass) {
        int tp = 0, fn = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == positiveClass) {
                if (predicted[i] == positiveClass) tp++; else fn++;
            }
        }
        return (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
    }

    public static double f1Score(int[] trueLabels, int[] predicted, int positiveClass) {
        double p = precision(trueLabels, predicted, positiveClass);
        double r = recall(trueLabels, predicted, positiveClass);
        return (p + r) == 0 ? 0.0 : 2 * p * r / (p + r);
    }

    public static double meanSquaredError(double[] trueValues, double[] predicted) {
        if (trueValues.length != predicted.length) {
            throw new IllegalArgumentException("Array length mismatch");
        }
        double sum = 0;
        for (int i = 0; i < trueValues.length; i++) {
            double diff = trueValues[i] - predicted[i];
            sum += diff * diff;
        }
        return sum / trueValues.length;
    }

    public static double meanAbsoluteError(double[] trueValues, double[] predicted) {
        if (trueValues.length != predicted.length) {
            throw new IllegalArgumentException("Array length mismatch");
        }
        double sum = 0;
        for (int i = 0; i < trueValues.length; i++) {
            sum += Math.abs(trueValues[i] - predicted[i]);
        }
        return sum / trueValues.length;
    }
}
