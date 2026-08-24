package tech.kayys.tafkir.ml.timeseries.data;

/**
 * Scales data to the [0, 1] range using observed min and max.
 *
 * <p>{@code x_scaled = (x - min) / (max - min)}
 */
public final class MinMaxScaler implements TimeSeriesScaler {

    private double min = Double.NaN;
    private double max = Double.NaN;

    @Override
    public double[] fitTransform(double[] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must not be empty");

        min = data[0]; max = data[0];
        for (double v : data) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return transform(data);
    }

    @Override
    public double[] transform(double[] data) {
        checkFitted();
        double range = max - min;
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (range < 1e-12) ? 0.0 : (data[i] - min) / range;
        }
        return result;
    }

    @Override
    public double[] inverseTransform(double[] scaled) {
        checkFitted();
        double range = max - min;
        double[] result = new double[scaled.length];
        for (int i = 0; i < scaled.length; i++) {
            result[i] = scaled[i] * range + min;
        }
        return result;
    }

    @Override
    public double inverseTransformScalar(double scaled) {
        checkFitted();
        return scaled * (max - min) + min;
    }

    public double min() { checkFitted(); return min; }
    public double max() { checkFitted(); return max; }

    @Override
    public String name() { return "MinMaxScaler"; }

    private void checkFitted() {
        if (Double.isNaN(min)) throw new IllegalStateException("Scaler not fitted. Call fitTransform first.");
    }
}
