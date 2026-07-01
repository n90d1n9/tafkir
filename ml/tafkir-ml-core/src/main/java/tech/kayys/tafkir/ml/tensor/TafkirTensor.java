package tech.kayys.tafkir.ml.tensor;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.Buffer;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;

/**
 * Tafkir-facing tensor API. Wraps Aljabr's {@link Tensor} with a PyTorch-like
 * convenience layer, including factory methods suitable for JBang scripts.
 *
 * <p>All computation delegates to the underlying Aljabr {@link ComputeBackend},
 * which uses Panama FFM off-heap memory and Vector API SIMD on CPU.
 *
 * <p>Implements {@link Tensor} directly so it can be passed to Aljabr's own
 * layers and backends without unwrapping.
 */
public final class TafkirTensor implements Tensor {

    private Tensor delegate;
    private final ComputeBackend backend;

    private TafkirTensor(Tensor delegate, ComputeBackend backend) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    // ── Factory methods (JBang-friendly) ────────────────────────────────────

    public static TafkirTensor of(float[] data, long... shape) {
        CpuBackend cpu = TafkirBackend.cpu();
        Shape s = shape.length == 0 ? new Shape(data.length) : new Shape(shape);
        long expected = s.numel();
        if (data.length != expected) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " does not match shape " + Arrays.toString(shape) +
                " (expected " + expected + " elements)");
        }

        Buffer buffer = new tech.kayys.aljabr.core.memory.CpuBuffer(DType.F32.memoryFootprintBytes(expected));
        MemorySegment seg = buffer.segment();
        for (int i = 0; i < data.length; i++) {
            seg.set(ValueLayout.JAVA_FLOAT, i * 4L, data[i]);
        }

        DefaultTensor tensor = new DefaultTensor(s, DType.F32, DeviceType.CPU, buffer, cpu);
        return new TafkirTensor(tensor, cpu);
    }

    public static TafkirTensor zeros(long... shape) {
        CpuBackend cpu = TafkirBackend.cpu();
        Shape s = new Shape(shape);
        long expected = s.numel();
        Buffer buffer = new tech.kayys.aljabr.core.memory.CpuBuffer(DType.F32.memoryFootprintBytes(expected));
        buffer.segment().fill((byte) 0);
        DefaultTensor tensor = new DefaultTensor(s, DType.F32, DeviceType.CPU, buffer, cpu);
        return new TafkirTensor(tensor, cpu);
    }

    public static TafkirTensor ones(long... shape) {
        CpuBackend cpu = TafkirBackend.cpu();
        Shape s = new Shape(shape);
        long expected = s.numel();
        Buffer buffer = new tech.kayys.aljabr.core.memory.CpuBuffer(DType.F32.memoryFootprintBytes(expected));
        MemorySegment seg = buffer.segment();
        for (long i = 0; i < expected; i++) {
            seg.set(ValueLayout.JAVA_FLOAT, i * 4L, 1.0f);
        }
        DefaultTensor tensor = new DefaultTensor(s, DType.F32, DeviceType.CPU, buffer, cpu);
        return new TafkirTensor(tensor, cpu);
    }

    public static TafkirTensor rand(long... shape) {
        CpuBackend cpu = TafkirBackend.cpu();
        Shape s = new Shape(shape);
        long expected = s.numel();
        Buffer buffer = new tech.kayys.aljabr.core.memory.CpuBuffer(DType.F32.memoryFootprintBytes(expected));
        MemorySegment seg = buffer.segment();
        java.util.concurrent.ThreadLocalRandom rng = java.util.concurrent.ThreadLocalRandom.current();
        for (long i = 0; i < expected; i++) {
            seg.set(ValueLayout.JAVA_FLOAT, i * 4L, rng.nextFloat());
        }
        DefaultTensor tensor = new DefaultTensor(s, DType.F32, DeviceType.CPU, buffer, cpu);
        return new TafkirTensor(tensor, cpu);
    }

    public static TafkirTensor randn(long... shape) {
        CpuBackend cpu = TafkirBackend.cpu();
        Shape s = new Shape(shape);
        long expected = s.numel();
        Buffer buffer = new tech.kayys.aljabr.core.memory.CpuBuffer(DType.F32.memoryFootprintBytes(expected));
        MemorySegment seg = buffer.segment();
        java.util.concurrent.ThreadLocalRandom rng = java.util.concurrent.ThreadLocalRandom.current();
        for (long i = 0; i < expected; i++) {
            seg.set(ValueLayout.JAVA_FLOAT, i * 4L, (float) rng.nextGaussian());
        }
        DefaultTensor tensor = new DefaultTensor(s, DType.F32, DeviceType.CPU, buffer, cpu);
        return new TafkirTensor(tensor, cpu);
    }

    public static TafkirTensor scalar(float value) {
        return of(new float[]{value});
    }

    public static TafkirTensor fromAljabr(Tensor tensor) {
        return new TafkirTensor(tensor, tensor.backend());
    }

    // ── Shape & Properties ───────────────────────────────────────────────────

    @Override
    public Shape shape() { return delegate.shape(); }

    public long[] shapeArray() { return delegate.shape().dims(); }

    @Override
    public DType dtype() { return delegate.dtype(); }

    @Override
    public DeviceType device() { return delegate.device(); }

    @Override
    public long numel() { return delegate.numel(); }

    @Override
    public float item() {
        if (delegate.numel() != 1) {
            throw new IllegalStateException(
                "Cannot call item() on tensor with " + delegate.numel() + " elements.");
        }
        return delegate.item();
    }

    // ── Data Access ──────────────────────────────────────────────────────────

    public float[] data() {
        return toFloatArray();
    }

    @Override
    public float[] toFloatArray() {
        long n = delegate.numel();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalStateException("Tensor too large for float[]: " + n + " elements");
        }
        float[] result = new float[(int) n];

        if (delegate instanceof DefaultTensor dt) {
            Buffer buffer = dt.buffer();
            MemorySegment seg = buffer.segment();
            for (int i = 0; i < result.length; i++) {
                result[i] = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
            }
        } else {
            // Unoptimized copy if backend doesn't provide toFloatArray natively
            float[] fallback = delegate.toFloatArray();
            if (fallback != null) {
                return fallback;
            }
            for (int i = 0; i < result.length; i++) {
                result[i] = 0.0f; // placeholder
            }
        }
        return result;
    }

    // ── In-place operations ────────────────────────────────────────────────

    public void fill_(float value) {
        if (delegate instanceof DefaultTensor dt) {
            MemorySegment seg = dt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, value);
            }
        } else {
            throw new UnsupportedOperationException("fill_ only supported for DefaultTensor");
        }
    }

    public void add_(TafkirTensor other) {
        if (delegate instanceof DefaultTensor dt && other.delegate instanceof DefaultTensor odt) {
            MemorySegment seg = dt.buffer().segment();
            MemorySegment oseg = odt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                float v = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                float ov = oseg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, v + ov);
            }
        } else {
            this.delegate = delegate.add(other.delegate);
        }
    }

    public void sub_(TafkirTensor other) {
        if (delegate instanceof DefaultTensor dt && other.delegate instanceof DefaultTensor odt) {
            MemorySegment seg = dt.buffer().segment();
            MemorySegment oseg = odt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                float v = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                float ov = oseg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, v - ov);
            }
        } else {
            this.delegate = delegate.sub(other.delegate);
        }
    }

    public void mul_(float scalar) {
        if (delegate instanceof DefaultTensor dt) {
            MemorySegment seg = dt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                float v = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, v * scalar);
            }
        } else {
            this.delegate = delegate.mul(scalar);
        }
    }

    public void div_(float scalar) {
        if (delegate instanceof DefaultTensor dt) {
            MemorySegment seg = dt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                float v = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, v / scalar);
            }
        } else {
            this.delegate = delegate.div(scalar);
        }
    }

    public void sqrt_() {
        if (delegate instanceof DefaultTensor dt) {
            MemorySegment seg = dt.buffer().segment();
            long numel = delegate.numel();
            for (long i = 0; i < numel; i++) {
                float v = seg.get(ValueLayout.JAVA_FLOAT, i * 4L);
                seg.set(ValueLayout.JAVA_FLOAT, i * 4L, (float) Math.sqrt(v));
            }
        } else {
            this.delegate = delegate.sqrt();
        }
    }

    // ── Gradients ────────────────────────────────────────────────────────────

    @Override
    public boolean requiresGrad() { return delegate.requiresGrad(); }

    public TafkirTensor requiresGrad(boolean requiresGrad) {
        delegate.setRequiresGrad(requiresGrad);
        return this;
    }

    @Override
    public Tensor grad() { return delegate.grad(); }

    public TafkirTensor gradTensor() {
        Tensor g = delegate.grad();
        return g == null ? null : new TafkirTensor(g, backend);
    }

    @Override
    public void setGrad(Tensor grad) { delegate.setGrad(grad); }

    @Override
    public void setRequiresGrad(boolean requiresGrad) { delegate.setRequiresGrad(requiresGrad); }

    @Override
    public void backward() { delegate.backward(); }

    // ── Tensor Operations ────────────────────────────────────────────────────

    @Override
    public Tensor add(Tensor t) { return wrap(delegate.add(t)); }
    public TafkirTensor add(TafkirTensor t) { return wrap(delegate.add(t.delegate)); }

    @Override
    public Tensor sub(Tensor t) { return wrap(delegate.sub(t)); }
    public TafkirTensor sub(TafkirTensor t) { return wrap(delegate.sub(t.delegate)); }

    @Override
    public Tensor mul(Tensor t) { return wrap(delegate.mul(t)); }
    public TafkirTensor mul(TafkirTensor t) { return wrap(delegate.mul(t.delegate)); }

    @Override
    public Tensor div(Tensor t) { return wrap(delegate.div(t)); }

    @Override
    public Tensor add(float scalar) { return wrap(delegate.add(scalar)); }
    @Override
    public Tensor mul(float scalar) { return wrap(delegate.mul(scalar)); }
    @Override
    public Tensor div(float scalar) { return wrap(delegate.div(scalar)); }

    @Override
    public Tensor matmul(Tensor t) { return wrap(delegate.matmul(t)); }
    public TafkirTensor matmul(TafkirTensor t) { return wrap(delegate.matmul(t.delegate)); }

    @Override
    public Tensor reshape(long... newShape) { return wrap(delegate.reshape(newShape)); }
    @Override
    public Tensor transpose() { return wrap(delegate.transpose()); }
    @Override
    public Tensor transpose(int dim0, int dim1) { return wrap(delegate.transpose(dim0, dim1)); }
    @Override
    public Tensor relu() { return wrap(delegate.relu()); }
    @Override
    public Tensor gelu() { return wrap(delegate.gelu()); }
    @Override
    public Tensor sigmoid() { return wrap(delegate.sigmoid()); }
    @Override
    public Tensor tanh() { return wrap(delegate.tanh()); }
    @Override
    public Tensor silu() { return wrap(delegate.silu()); }
    @Override
    public Tensor softmax() { return wrap(delegate.softmax()); }
    @Override
    public Tensor softmax(int dim) { return wrap(delegate.softmax(dim)); }
    @Override
    public Tensor logSoftmax(int dim) { return wrap(delegate.logSoftmax(dim)); }
    @Override
    public Tensor mean() { return wrap(delegate.mean()); }
    @Override
    public Tensor mean(int dim, boolean keepDim) { return wrap(delegate.mean(dim, keepDim)); }
    @Override
    public Tensor sum() { return wrap(delegate.sum()); }
    @Override
    public Tensor sum(int dim, boolean keepDim) { return wrap(delegate.sum(dim, keepDim)); }
    @Override
    public Tensor pow(float exponent) { return wrap(delegate.pow(exponent)); }
    @Override
    public Tensor sqrt() { return wrap(delegate.sqrt()); }
    @Override
    public Tensor abs() { return wrap(delegate.abs()); }
    @Override
    public Tensor exp() { return wrap(delegate.exp()); }
    @Override
    public Tensor log() { return wrap(delegate.log()); }
    @Override
    public Tensor flatten() { return wrap(delegate.flatten()); }
    @Override
    public Tensor unsqueeze(int dim) { return wrap(delegate.unsqueeze(dim)); }
    @Override
    public Tensor squeeze() { return wrap(delegate.squeeze()); }
    @Override
    public Tensor crossEntropy(Tensor target) { return wrap(delegate.crossEntropy(target)); }
    @Override
    public Tensor binaryCrossEntropy(Tensor target) { return wrap(delegate.binaryCrossEntropy(target)); }
    @Override
    public Tensor dropout(float p, boolean training) { return wrap(delegate.dropout(p, training)); }
    @Override
    public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        return wrap(delegate.layerNorm(normalizedShape, weight, bias, eps));
    }
    @Override
    public Tensor rmsNorm(Tensor weight, float eps) { return wrap(delegate.rmsNorm(weight, eps)); }
    @Override
    public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training, float momentum, float eps) {
        return wrap(delegate.batchNorm(weight, bias, runningMean, runningVar, training, momentum, eps));
    }
    @Override
    public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return wrap(delegate.conv2d(weight, bias, stride, padding, dilation, groups));
    }
    @Override
    public Tensor maxPool2d(int kernelSize, int stride, int padding) {
        return wrap(delegate.maxPool2d(kernelSize, stride, padding));
    }
    @Override
    public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
        return wrap(delegate.adaptiveAvgPool2d(outputH, outputW));
    }
    @Override
    public Tensor attention(Tensor K, Tensor V) { return wrap(delegate.attention(K, V)); }
    @Override
    public Tensor embedding(Tensor weight, long paddingIdx) {
        return wrap(delegate.embedding(weight, paddingIdx));
    }
    @Override
    public Tensor slice(long[] offsets, long[] sizes) { return wrap(delegate.slice(offsets, sizes)); }
    @Override
    public java.util.List<Tensor> split(int axis, int parts) { return delegate.split(axis, parts); }
    @Override
    public Tensor cast(DType dtype) { return wrap(delegate.cast(dtype)); }
    @Override
    public Tensor to(DeviceType device) { return wrap(delegate.to(device)); }
    @Override
    public Tensor zerosLike() { return wrap(delegate.zerosLike()); }
    @Override
    public Tensor eval() { return wrap(delegate.eval()); }

    // ── Backend & Internal ─────────────────────────────────────────────────

    @Override
    public ComputeBackend backend() { return delegate.backend(); }

    public Tensor unwrap() { return delegate; }

    private TafkirTensor wrap(Tensor t) {
        if (t == delegate) return this;
        return new TafkirTensor(t, backend);
    }

    @Override
    public String toString() {
        return "TafkirTensor{shape=" + Arrays.toString(shapeArray()) +
               ", dtype=" + dtype() + ", device=" + device() +
               ", requiresGrad=" + requiresGrad() + ", numel=" + numel() + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TafkirTensor other)) return false;
        return delegate.equals(other.delegate);
    }

    @Override
    public int hashCode() { return delegate.hashCode(); }
}
