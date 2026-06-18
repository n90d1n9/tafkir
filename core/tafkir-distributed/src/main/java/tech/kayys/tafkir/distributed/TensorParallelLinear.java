package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public final class TensorParallelLinear {
    private final CommBackend comm;

    public TensorParallelLinear(CommBackend comm) {
        this.comm = comm;
    }

    public Tensor forward(Tensor x, Tensor localWeight) {
        // partial output
        Tensor out = x.matmul(localWeight);
        // gather all parts
        return comm.allGather(out);
    }
}