package tech.kayys.aljabr.safetensor.loader;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Stub — FFM-backed loader for safetensor files.
 * Real implementation uses memory-mapped I/O via JDK Foreign Function Memory API.
 */
public class SafetensorFFMLoader {

    public SafetensorFFMLoader() {}

    /**
     * Opens a safetensor file and returns a {@link SafetensorLoadResult}.
     * Caller is responsible for closing the result.
     *
     * @param path path to a {@code .safetensors} file
     * @return loaded result
     * @throws IOException if the file cannot be read or parsed
     */
    public SafetensorLoadResult load(Path path) throws IOException {
        throw new UnsupportedOperationException(
                "SafetensorFFMLoader stub — real implementation not yet linked. "
                + "Path requested: " + path);
    }
}
