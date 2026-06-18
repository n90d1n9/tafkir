package tech.kayys.tafkir.training.strategy.ffi;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * FFM bindings specifically for executing Tafkir training operations via LibTorch.
 * This directly accesses the off-heap MemorySegments provided by Aljabr UnifiedMemoryStore.
 */
public class TafkirLibTorchBinding {

    private static final Linker LINKER = Linker.nativeLinker();
    private static volatile TafkirLibTorchBinding instance;

    private final SymbolLookup lookup;
    private final ConcurrentMap<String, Optional<MethodHandle>> handleCache = new ConcurrentHashMap<>();

    // ── TorchTensor creation ───────────────────────────────────────────────

    /** at::from_blob(void* data, IntArrayRef size, ScalarType dtype) → TorchTensor */
    public static final String TENSOR_FROM_BLOB = "at_from_blob";
    public static final FunctionDescriptor TENSOR_FROM_BLOB_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT);

    // ── Autograd / Backward Pass ───────────────────────────────────────────────

    /** tensor.requires_grad_(bool) → TorchTensor */
    public static final String TENSOR_REQUIRES_GRAD = "at_requires_grad_";
    public static final FunctionDescriptor TENSOR_REQUIRES_GRAD_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN);

    /** tensor.backward() */
    public static final String TENSOR_BACKWARD = "at_backward";
    public static final FunctionDescriptor TENSOR_BACKWARD_DESC = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);

    /** tensor.grad() → TorchTensor */
    public static final String TENSOR_GRAD = "at_grad";
    public static final FunctionDescriptor TENSOR_GRAD_DESC = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    // ── Optimization ─────────────────────────────────────────────────────────

    /** adam_step(param, grad, exp_avg, exp_avg_sq, ...) */
    public static final String ADAM_STEP = "at_adam_step";
    public static final FunctionDescriptor ADAM_STEP_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
            ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_BOOLEAN);

    // ── Forward Pass Ops (LoRA specific) ─────────────────────────────────────

    /** tensor.matmul(other) → TorchTensor */
    public static final String TENSOR_MATMUL = "at_matmul";
    public static final FunctionDescriptor TENSOR_MATMUL_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    public static final String TENSOR_ADD = "at_add";
    public static final FunctionDescriptor TENSOR_ADD_DESC = FunctionDescriptor.of(
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    private TafkirLibTorchBinding(SymbolLookup lookup) {
        this.lookup = lookup;
    }

    public static void initialize(SymbolLookup lookup) {
        if (instance == null) {
            synchronized (TafkirLibTorchBinding.class) {
                if (instance == null) {
                    instance = new TafkirLibTorchBinding(lookup);
                }
            }
        }
    }

    public static TafkirLibTorchBinding getInstance() {
        if (instance == null) {
            // For environments where LibTorch is not yet loaded, return an empty binding lookup gracefully
            // or rely on a default System lookup if available. 
            // In a real environment, this must be initialized by the engine bootstrapper.
            throw new IllegalStateException("TafkirLibTorchBinding is not initialized.");
        }
        return instance;
    }

    public Optional<MethodHandle> bindOptional(String name, FunctionDescriptor descriptor) {
        return handleCache.computeIfAbsent(name, k -> {
            try {
                Optional<MemorySegment> symbol = lookup.find(k);
                if (symbol.isPresent()) {
                    return Optional.of(LINKER.downcallHandle(symbol.get(), descriptor));
                }
                return Optional.empty();
            } catch (Exception e) {
                return Optional.empty();
            }
        });
    }

    public MethodHandle bind(String name, FunctionDescriptor descriptor) {
        return bindOptional(name, descriptor).orElseThrow(() -> new UnsatisfiedLinkError("Missing native LibTorch symbol: " + name));
    }
}
