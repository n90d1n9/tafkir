package tech.kayys.tafkir.ml.timeseries.data;

/**
 * SPI for reversible scaling of univariate time-series data.
 */
public interface TimeSeriesScaler {

    /**
     * Fit the scaler to {@code data} and return a scaled copy.
     * Calling {@link #transform(double[])} after this reuses the fitted statistics.
     */
    double[] fitTransform(double[] data);

    /** Scale {@code data} using previously fitted statistics. */
    double[] transform(double[] data);

    /** Invert scaling to recover original-scale values. */
    double[] inverseTransform(double[] scaled);

    /** Invert a single scaled point. */
    double inverseTransformScalar(double scaled);

    /** Human-readable name. */
    String name();
}
