package tech.kayys.tafkir.ml;

/**
 * @deprecated Use {@link Aljabr} as the unified ML entry point.
 *             {@code AljabrML} remains as a thin source-compatible shim for
 *             existing code while all implementation lives in {@link Aljabr}.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class AljabrML {
    public static class DL extends Aljabr.DL {
    }

    public static class ML extends Aljabr.ML {
    }

    public static class Selection extends Aljabr.Selection {
    }

    public static class Hub extends Aljabr.Hub {
    }

    public static class Export extends Aljabr.Export {
    }
}
