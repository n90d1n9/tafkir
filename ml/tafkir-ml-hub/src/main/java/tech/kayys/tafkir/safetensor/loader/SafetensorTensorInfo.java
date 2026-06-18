package tech.kayys.aljabr.safetensor.loader;

/**
 * Stub — metadata for a single tensor entry in a safetensor file header.
 */
public final class SafetensorTensorInfo {

    private final String name;
    private final String rawDtype;
    private final long[] shape;
    private final long[] dataOffsets; // [begin, end)

    public SafetensorTensorInfo(String name, String rawDtype, long[] shape, long[] dataOffsets) {
        this.name = name;
        this.rawDtype = rawDtype;
        this.shape = shape;
        this.dataOffsets = dataOffsets;
    }

    public String name() { return name; }
    public long[] shape() { return shape; }
    public long[] dataOffsets() { return dataOffsets; }

    /** Parsed dtype. Returns {@code null} if unrecognized. */
    public SafetensorDType dtype() {
        try {
            return SafetensorDType.valueOf(rawDtype);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    public String rawDtype() { return rawDtype; }
}
