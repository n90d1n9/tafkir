package tech.kayys.tafkir.ml.export.litert;

import tech.kayys.tafkir.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Stub LiteRT (TFLite) exporter.
 * Real implementation lives in {@code runner:litert:aljabr-litert-core} (currently disabled).
 */
public final class LiteRTExporter {

    private final NNModule model;
    private final long[] inputShape;

    private LiteRTExporter(NNModule model, long[] inputShape) {
        this.model = model;
        this.inputShape = inputShape;
    }

    public static LiteRTExporter fromModel(NNModule model, long[] inputShape) {
        return new LiteRTExporter(model, inputShape);
    }

    public void export(Path outputPath) throws IOException {
        throw new UnsupportedOperationException(
                "LiteRTExporter is a stub — enable :runner:litert:aljabr-litert-core for full support");
    }
}
