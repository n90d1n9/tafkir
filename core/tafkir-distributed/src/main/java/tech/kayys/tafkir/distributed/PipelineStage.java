package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public interface PipelineStage {
    Tensor forward(Tensor input);

    Tensor backward(Tensor grad);
}