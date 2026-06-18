package tech.kayys.aljabr.safetensor.loader;

import java.lang.foreign.MemorySegment;

/**
 * Stub — represents a single named tensor from a safetensor shard file.
 */
public interface SafetensorTensor {
    String name();
    SafetensorDType dtype();
    long[] shape();
    /** Memory-mapped segment covering this tensor's data. */
    MemorySegment segment();
    float[] toFloatArray();
}

