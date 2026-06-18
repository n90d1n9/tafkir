package tech.kayys.aljabr.gguf.core;

import tech.kayys.tafkir.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Stub GGUF exporter.
 * Real implementation lives in {@code runner:gguf:aljabr-gguf-core} (currently disabled).
 */
public final class GgufExporter {

    private final NNModule model;
    private final Map<String, Object> metadata;
    private Quantization quantization = Quantization.NONE;

    private GgufExporter(NNModule model, Map<String, Object> metadata) {
        this.model = model;
        this.metadata = metadata;
    }

    public static GgufExporter fromModel(NNModule model, Map<String, Object> metadata) {
        return new GgufExporter(model, metadata);
    }

    public GgufExporter quantization(Quantization q) {
        this.quantization = q;
        return this;
    }

    public void export(Path outputPath) throws IOException {
        throw new UnsupportedOperationException(
                "GgufExporter is a stub — enable :runner:gguf:aljabr-gguf-core for full support");
    }

    public enum Quantization {
        NONE, FP16, INT8, INT4, NF4
    }
}
