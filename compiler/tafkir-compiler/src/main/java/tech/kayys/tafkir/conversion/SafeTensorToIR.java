package tech.kayys.tafkir.conversion;

import tech.kayys.tafkir.ir.*;
import tech.kayys.tafkir.core.tensor.*;
import tech.kayys.tafkir.core.tensor.ModelWeightLoader;
import tech.kayys.tafkir.core.tensor.WeightAdapter;
import java.nio.file.Path;
import java.util.*;

public final class SafeTensorToIR implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".safetensors");
    }

    @Override
    public WeightAdapter load(Path path) {
        // TODO: Implement actual loading
        throw new UnsupportedOperationException("SafeTensorToIR not implemented yet");
    }
}
