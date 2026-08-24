package tech.kayys.tafkir.ml.timeseries.data;

/**
 * Standardises data to zero mean and unit variance (z-score normalisation).
 *
 * <p>{@code x_scaled = (x - mean) / std}
 */
public final class StandardScaler implements TimeSeriesScaler {

    private double mean = Double.NaN;
    private double std  = Double.NaN;

    @Override
    public double[] fitTransform(double[] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must not be empty");

        double sum = 0.0;
        for (double v : data) sum += v;
        mean = sum / data.length;

        double variance = 0.0;
        for (double v : data) variance += (v - mean) * (v - mean);
        std = Math.sqrt(variance / data.length);

        return transform(data);
    }

    @Override
    public double[] transform(double[] data) {
        checkFitted();
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (std < 1e-12) ? 0.0 : (data[i] - mean) / std;
        }
        return result;
    }

    @Override
    public double[] inverseTransform(double[] scaled) {
        checkFitted();
        double[] result = new double[scaled.length];
        for (int i = 0; i < scaled.length; i++) {
            result[i] = scaled[i] * std + mean;
        }
        return result;
    }

    @Override
    public double inverseTransformScalar(double scaled) {
        checkFitted();
        return scaled * std + mean;
    }

    public double mean() { checkFitted(); return mean; }
    public double std()  { checkFitted(); return std; }

    @Override
    public String name() { return "StandardScaler"; }

    private void checkFitted() {
        if (Double.isNaN(mean)) throw new IllegalStateException("Scaler not fitted. Call fitTransform first.");
    }
}
