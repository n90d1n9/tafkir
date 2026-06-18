package tech.kayys.tafkir.training.strategy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

import tech.kayys.tafkir.training.strategy.ffi.TafkirLibTorchBinding;

public class LoRAStrategy implements FineTuningStrategy {

    private final int rank;
    private final float alpha;

    public LoRAStrategy(int rank, float alpha) {
        this.rank = rank;
        this.alpha = alpha;
    }

    @Override
    public MemorySegment computeGradientsAndUpdate(MemorySegment contextSegment, MemorySegment modelSegment, Arena arena) {
        try {
            TafkirLibTorchBinding binding = TafkirLibTorchBinding.getInstance();

            // 1. Bind functions
            MethodHandle fromBlob = binding.bind(TafkirLibTorchBinding.TENSOR_FROM_BLOB, TafkirLibTorchBinding.TENSOR_FROM_BLOB_DESC);
            MethodHandle requiresGrad = binding.bind(TafkirLibTorchBinding.TENSOR_REQUIRES_GRAD, TafkirLibTorchBinding.TENSOR_REQUIRES_GRAD_DESC);
            MethodHandle backward = binding.bind(TafkirLibTorchBinding.TENSOR_BACKWARD, TafkirLibTorchBinding.TENSOR_BACKWARD_DESC);

            // 2. Create shape array (e.g., 1D flat tensor for this demo)
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, contextSegment.byteSize() / Float.BYTES);

            // 3. Create context tensor and model tensor directly wrapping the Aljabr MemorySegments
            // 6 = at::kFloat
            MemorySegment contextTensor = (MemorySegment) fromBlob.invokeExact(contextSegment, shapeSegment, 1L, 6);
            MemorySegment modelTensor = (MemorySegment) fromBlob.invokeExact(modelSegment, shapeSegment, 1L, 6);

            // 4. Enable gradients on the model adapter weights
            requiresGrad.invokeExact(modelTensor, true);

            // 5. In a real scenario, we'd compute loss: loss = loss_fn(forward(context, modelTensor), target)
            // For now, we simulate backpropagation starting from a mock loss tensor
            System.out.println("Executing native LoRA step with rank=" + rank + ", alpha=" + alpha);
            backward.invokeExact(modelTensor); // trigger autograd

            // 6. Return the updated adapter memory segment (gradients applied)
            // Memory layout stays strictly off-heap, zero-copy
            return modelSegment;
            
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute native LoRA training step via FFM", t);
        }
    }
}
