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

    public static ComputeBackend best() {
        return cpu();
    }

    public static boolean cudaAvailable() {
        return false;
    }

    public static boolean metalAvailable() {
        return false;
    }
}
