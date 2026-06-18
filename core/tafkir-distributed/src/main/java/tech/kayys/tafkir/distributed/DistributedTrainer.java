package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;
import tech.kayys.tafkir.ir.*;
import java.util.Map;

/**
 * Distributed trainer - requires tafkir-train module at runtime.
 * Stubbed for compilation.
 */
public final class DistributedTrainer {
    private final Object base; // Trainer - from training module
    private final DDPWrapper ddp;

    public DistributedTrainer(Object base, DDPWrapper ddp) {
        this.base = base;
        this.ddp = ddp;
    }

    public void trainStep(GGraph model,
            GValueId loss,
            Map<String, Tensor> params,
            Map<String, Tensor> inputs) {
        throw new UnsupportedOperationException(
                "DistributedTrainer requires tafkir-train module");
    }
}