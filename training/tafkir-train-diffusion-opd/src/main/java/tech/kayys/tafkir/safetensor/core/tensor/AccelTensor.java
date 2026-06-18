package tech.kayys.aljabr.safetensor.core.tensor;

/**
 * Stub — accelerator-native tensor used by the safetensor Stable Diffusion runner.
 *
 * <p>This is a bridge transport type; the real implementation is in the safetensor runner module.
 * Close via try-with-resources to release native memory.
 */
public class AccelTensor implements AutoCloseable {

    private final float[] data;
    private final long[] shape;

    private AccelTensor(float[] data, long[] shape) {
        this.data = data;
        this.shape = shape;
    }

    /**
     * Create an AccelTensor from a float array and a shape (dims).
     */
    public static AccelTensor fromFloatArray(float[] values, long[] dims) {
        return new AccelTensor(values.clone(), dims.clone());
    }

    /** Return a copy of this tensor's data as a float array. */
    public float[] toFloatArray() {
        return data.clone();
    }

    /**
     * Shape dims — returns the raw long[] shape array directly,
     * compatible with adapter code that passes this to {@code long...} varargs.
     */
    public long[] shape() {
        return shape.clone();
    }

    @Override
    public void close() {
        // Stub: no native memory to free
    }
}
