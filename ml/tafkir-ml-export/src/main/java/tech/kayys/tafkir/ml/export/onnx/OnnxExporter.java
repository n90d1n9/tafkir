package tech.kayys.tafkir.ml.export.onnx;

import tech.kayys.tafkir.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Stub ONNX exporter.
 * Real implementation lives in {@code runner:onnx:tafkir-ml-export-onnx} (currently disabled).
 */
public final class OnnxExporter {

    private final NNModule model;
    private final long[] inputShape;

    private OnnxExporter(NNModule model, long[] inputShape) {
        this.model = model;
        this.inputShape = inputShape;
    }

    public static OnnxExporter fromModel(NNModule model, long[] inputShape) {
        return new OnnxExporter(model, inputShape);
    }

    public void export(Path outputPath) throws IOException {
        throw new UnsupportedOperationException(
                "OnnxExporter is a stub — enable :runner:onnx:tafkir-ml-export-onnx for full support");
    }
}
