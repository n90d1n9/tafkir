package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public interface CommBackend {
    Tensor allReduce(Tensor t);

    Tensor broadcast(Tensor t, int src);

    Tensor allGather(Tensor t);

    void barrier();
}