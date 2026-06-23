package tech.kayys.tafkir.ml.tensor;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;

/**
 * Provides access to Aljabr compute backends.
 * Detects available hardware and returns the best backend.
 */
public final class TafkirBackend {

    private static CpuBackend CPU;

    private TafkirBackend() {}

    public static synchronized CpuBackend cpu() {
        if (CPU == null) {
            CPU = new CpuBackend();
        }
        return CPU;
    }

    /**
     * Returns the best available backend.
     * Currently only CPU is supported for training.
     * CUDA and Metal are available for inference via the CLI.
     */
    public static ComputeBackend best() {
        return cpu();
    }

    /**
     * Check if CUDA is available.
     */
    public static boolean cudaAvailable() {
        // Will be implemented when CUDA training backend is ready
        return false;
    }

    /**
     * Check if Metal is available (Apple Silicon).
     */
    public static boolean metalAvailable() {
        // Will be implemented when Metal training backend is ready
        return false;
    }
}
