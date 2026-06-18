package tech.kayys.aljabr.safetensor.loader;

import java.lang.foreign.MemorySegment;
import java.util.Set;

/**
 * Stub — result of loading a single safetensor shard file.
 * Real implementation lives in the safetensor-loader module.
 */
public interface SafetensorLoadResult extends AutoCloseable {

    /** Access a tensor by name. Returns null if not found. */
    SafetensorTensor tensor(String name);

    /** Return the parsed file header. */
    SafetensorHeader header();

    /** Return the set of tensor names. */
    Set<String> tensorNames();

    @Override
    void close();
}

