package tech.kayys.tafkir.conversion;

import tech.kayys.tafkir.ir.*;
import tech.kayys.tafkir.core.tensor.ModelWeightLoader;
import tech.kayys.tafkir.core.tensor.WeightAdapter;
import java.nio.file.Path;

public final class OnnxToIR implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".onnx");
    }

    @Override
    public WeightAdapter load(Path path) {
        throw new UnsupportedOperationException("OnnxToIR not implemented");
    }
}