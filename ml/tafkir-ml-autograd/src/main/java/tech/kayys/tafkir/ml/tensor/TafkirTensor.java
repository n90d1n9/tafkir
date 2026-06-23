package tech.kayys.tafkir.ml.tensor;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.memory.CpuBuffer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Tafkir-facing tensor API that delegates all computation to Aljabr backends.
 *
 * <p>This is a thin wrapper around Aljabr's {@code Tensor} and {@code ComputeBackend}
 * abstractions. It provides a PyTorch-like API while leveraging Aljabr's optimized
 * implementations including:</p>
 *
 * <ul>
 *   <li>Vector API SIMD acceleration on CPU</li>
 *   <li>Panama FFM off-heap memory management</li>
 *   <li>Graph-based autograd engine</li>
 *   <li>Multi-backend support (CPU, CUDA, Metal, Blackwell)</li>
 * </ul>
 *
 * <p><b>Architecture note:</b> Tafkir is a compatibility layer over Aljabr.
 * All tensor operations delegate to Aljabr's {@code ComputeBackend} implementation.</p>
 *
 * @see Tensor
 * @see CpuBackend
 */
public final class TafkirTensor {

    private final Tensor delegate;
    private final CpuBackend backend;

    private TafkirTensor(Tensor delegate, CpuBackend backend) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * Creates a tensor from raw float data.
     *
     * @param data  the float array (will be copied)
     * @param shape the tensor shape
     * @return a new TafkirTensor backed by Aljabr's Tensor
     */
    public static TafkirTensor of(float[] data, long... shape) {
        CpuBackend backend = new CpuBackend();
        Shape s = Shape.of(shape);
        Tensor t = backend.create(s, DType.F32, data);
        return new TafkirTensor(t, backend);
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * @param shape the tensor shape
     * @return a new TafkirTensor filled with zeros
     */
    public static TafkirTensor zeros(long... shape) {
        CpuBackend backend = new CpuBackend();
        Tensor t = backend.zeros(Shape.of(shape));
        return new TafkirTensor(t, backend);
    }

    /**
     * Creates a tensor filled with ones.
     *
     * @param shape the tensor shape
     * @return a new TafkirTensor filled with ones
     */
    public static TafkirTensor ones(long... shape) {
        CpuBackend backend = new CpuBackend();
        Tensor t = backend.ones(Shape.of(shape));
        return new TafkirTensor(t, backend);
    }

    /**
     * Creates a tensor with random values from a normal distribution.
     *
     * @param shape the tensor shape
     * @return a new TafkirTensor with.randn values
     */
    public static TafkirTensor randn(long... shape) {
        CpuBackend backend = new CpuBackend();
        Tensor t = backend.randn(Shape.of(shape));
        return new TafkirTensor(t, backend);
    }

    /**
     * Creates a tensor with random values from a uniform distribution.
     *
     * @param low   lower bound (inclusive)
     * @param high  upper bound (exclusive)
     * @param shape the tensor shape
     * @return a new TafkirTensor with uniform random values
     */
    public static TafkirTensor rand(double low, double high, long... shape) {
        CpuBackend backend = new CpuBackend();
        Tensor t = backend.rand(Shape.of(shape), (float) low, (float) high);
        return new TafkirTensor(t, backend);
    }

    /**
     * Creates a scalar tensor.
     *
     * @param value the scalar value
     * @return a new TafkirTensor with shape []
     */
    public static TafkirTensor scalar(float value) {
        return of(new float[]{value});
    }

    // =========================================================================
    // Basic Properties
    // =========================================================================

    /**
     * Returns the tensor's shape.
     *
     * @return a copy of the shape array
     */
    public long[] shape() {
        try {
            return (long[]) invokeShape(delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get shape", e);
        }
    }

    /**
     * Returns the number of dimensions (rank).
     *
     * @return the rank of this tensor
     */
    public int ndim() {
        return shape().length;
    }

    /**
     * Returns the total number of elements.
     *
     * @return the number of elements
     */
    public long numel() {
        long[] s = shape();
        long n = 1;
        for (long dim : s) {
            n *= dim;
        }
        return n;
    }

    /**
     * Returns whether this tensor requires gradient computation.
     *
     * @return true if gradients should be computed
     */
    public boolean requiresGrad() {
        try {
            return (Boolean) invokeRequiresGrad(delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to check requiresGrad", e);
        }
    }

    /**
     * Sets whether this tensor requires gradient computation.
     *
     * @param requiresGrad true to enable gradient tracking
     * @return this tensor for chaining
     */
    public TafkirTensor requiresGrad(boolean requiresGrad) {
        try {
            invokeSetRequiresGrad(delegate, requiresGrad);
            return this;
        } catch (Exception e) {
            throw new RuntimeException("Failed to set requiresGrad", e);
        }
    }

    /**
     * Returns the gradient tensor, or null if no gradient has been computed.
     *
     * @return the gradient tensor or null
     */
    public TafkirTensor grad() {
        try {
            Object gradDelegate = invokeGrad(delegate);
            if (gradDelegate == null) {
                return null;
            }
            return new TafkirTensor(gradDelegate, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get grad", e);
        }
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    /**
     * Copies the tensor data to a float array.
     *
     * @return a new float array containing the tensor's data
     */
    public float[] data() {
        try {
            return (float[]) invokeGetData(delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get data", e);
        }
    }

    /**
     * Returns the scalar value of a 0-dimensional tensor.
     *
     * @return the scalar value
     * @throws IllegalStateException if this tensor is not a scalar
     */
    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() can only be called on scalar tensors");
        }
        return data()[0];
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /**
     * Returns a new tensor with the negated values.
     *
     * @return -this
     */
    public TafkirTensor neg() {
        try {
            Object result = invokeBackendNeg(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to negate tensor", e);
        }
    }

    /**
     * Returns a new tensor with the absolute values.
     *
     * @return |this|
     */
    public TafkirTensor abs() {
        try {
            Object result = invokeBackendAbs(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute abs", e);
        }
    }

    /**
     * Returns a new tensor with the exponential of each element.
     *
     * @return exp(this)
     */
    public TafkirTensor exp() {
        try {
            Object result = invokeBackendExp(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute exp", e);
        }
    }

    /**
     * Returns a new tensor with the natural logarithm of each element.
     *
     * @return log(this)
     */
    public TafkirTensor log() {
        try {
            Object result = invokeBackendLog(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute log", e);
        }
    }

    /**
     * Returns a new tensor with the square root of each element.
     *
     * @return sqrt(this)
     */
    public TafkirTensor sqrt() {
        try {
            Object result = invokeBackendSqrt(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute sqrt", e);
        }
    }

    /**
     * Returns a new tensor with values clamped to the specified range.
     *
     * @param min minimum value
     * @param max maximum value
     * @return clamped tensor
     */
    public TafkirTensor clamp(float min, float max) {
        try {
            Object result = invokeBackendClamp(backend, delegate, min, max);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to clamp tensor", e);
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /**
     * Element-wise addition.
     *
     * @param other the tensor to add
     * @return this + other
     */
    public TafkirTensor add(TafkirTensor other) {
        try {
            Object result = invokeBackendAdd(backend, delegate, other.delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to add tensors", e);
        }
    }

    /**
     * Element-wise subtraction.
     *
     * @param other the tensor to subtract
     * @return this - other
     */
    public TafkirTensor sub(TafkirTensor other) {
        try {
            Object result = invokeBackendSub(backend, delegate, other.delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to subtract tensors", e);
        }
    }

    /**
     * Element-wise multiplication.
     *
     * @param other the tensor to multiply with
     * @return this * other (element-wise)
     */
    public TafkirTensor mul(TafkirTensor other) {
        try {
            Object result = invokeBackendMul(backend, delegate, other.delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to multiply tensors", e);
        }
    }

    /**
     * Element-wise division.
     *
     * @param other the tensor to divide by
     * @return this / other (element-wise)
     */
    public TafkirTensor div(TafkirTensor other) {
        try {
            Object result = invokeBackendDiv(backend, delegate, other.delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to divide tensors", e);
        }
    }

    /**
     * Matrix multiplication.
     *
     * @param other the tensor to multiply with
     * @return this @ other (matrix multiplication)
     */
    public TafkirTensor matmul(TafkirTensor other) {
        try {
            Object result = invokeBackendMatmul(backend, delegate, other.delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to matmul tensors", e);
        }
    }

    /**
     * Scalar multiplication.
     *
     * @param scalar the scalar value
     * @return this * scalar
     */
    public TafkirTensor mul(float scalar) {
        try {
            Object result = invokeBackendMulScalar(backend, delegate, scalar);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to multiply by scalar", e);
        }
    }

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    /**
     * Returns the sum of all elements.
     *
     * @return sum of elements as a scalar tensor
     */
    public TafkirTensor sum() {
        try {
            Object result = invokeBackendSum(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute sum", e);
        }
    }

    /**
     * Returns the mean of all elements.
     *
     * @return mean of elements as a scalar tensor
     */
    public TafkirTensor mean() {
        try {
            Object result = invokeBackendMean(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute mean", e);
        }
    }

    /**
     * Returns the maximum element.
     *
     * @return maximum element as a scalar tensor
     */
    public TafkirTensor max() {
        try {
            Object result = invokeBackendMax(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute max", e);
        }
    }

    /**
     * Returns the minimum element.
     *
     * @return minimum element as a scalar tensor
     */
    public TafkirTensor min() {
        try {
            Object result = invokeBackendMin(backend, delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to compute min", e);
        }
    }

    // =========================================================================
    // Shape Operations
    // =========================================================================

    /**
     * Returns a new tensor with the same data but different shape.
     *
     * @param newShape the new shape
     * @return reshaped tensor
     */
    public TafkirTensor reshape(long... newShape) {
        try {
            Object result = invokeReshape(delegate, newShape);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to reshape tensor", e);
        }
    }

    /**
     * Returns a view of this tensor with the given dimension removed (must be size 1).
     *
     * @param dim the dimension to squeeze
     * @return squeezed tensor
     */
    public TafkirTensor squeeze(int dim) {
        try {
            Object result = invokeSqueeze(delegate, dim);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to squeeze tensor", e);
        }
    }

    /**
     * Returns a view of this tensor with a new dimension inserted.
     *
     * @param dim the dimension to insert
     * @return tensor with new dimension
     */
    public TafkirTensor unsqueeze(int dim) {
        try {
            Object result = invokeUnsqueeze(delegate, dim);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to unsqueeze tensor", e);
        }
    }

    /**
     * Returns a transposed view of this tensor.
     *
     * @param dim0 first dimension
     * @param dim1 second dimension
     * @return transposed tensor
     */
    public TafkirTensor transpose(int dim0, int dim1) {
        try {
            Object result = invokeTranspose(delegate, dim0, dim1);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to transpose tensor", e);
        }
    }

    /**
     * Permutes the dimensions of this tensor.
     *
     * @param dims the new order of dimensions
     * @return permuted tensor
     */
    public TafkirTensor permute(int... dims) {
        try {
            Object result = invokePermute(delegate, dims);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to permute tensor", e);
        }
    }

    // =========================================================================
    // Autograd
    // =========================================================================

    /**
     * Computes gradients via backpropagation.
     *
     * <p>This triggers Aljabr's graph-based autograd engine to compute
     * gradients for all tensors with {@code requiresGrad=true} in the
     * computational graph leading to this tensor.</p>
     */
    public void backward() {
        try {
            invokeBackward(delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to run backward", e);
        }
    }

    /**
     * Computes gradients with a custom upstream gradient.
     *
     * @param gradOutput the upstream gradient tensor
     */
    public void backward(TafkirTensor gradOutput) {
        try {
            invokeBackwardWithGrad(delegate, gradOutput.delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to run backward", e);
        }
    }

    /**
     * Zeros out the gradient tensor.
     *
     * <p>Call this before each training step to accumulate fresh gradients.</p>
     */
    public void zeroGrad() {
        try {
            invokeZeroGrad(delegate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to zero grad", e);
        }
    }

    /**
     * Detaches this tensor from the computation graph.
     *
     * <p>The returned tensor shares the same data but won't track gradients.</p>
     *
     * @return detached tensor
     */
    public TafkirTensor detach() {
        try {
            Object result = invokeDetach(delegate);
            return new TafkirTensor(result, backend);
        } catch (Exception e) {
            throw new RuntimeException("Failed to detach tensor", e);
        }
    }

    // =========================================================================
    // Backend Access (for advanced use)
    // =========================================================================

    /**
     * Returns the underlying Aljabr Tensor.
     *
     * <p><b>Warning:</b> This exposes internal implementation details.
     * Use with caution.</p>
     *
     * @return the delegate Aljabr Tensor
     */
    public Object getDelegate() {
        return delegate;
    }

    // =========================================================================
    // Private Reflection Helpers (until Aljabr is directly linked)
    // =========================================================================

    private static Object createCpuBackend() throws Exception {
        Class<?> clazz = Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend");
        return clazz.getDeclaredConstructor().newInstance();
    }

    private static Object invokeBackendCreate(Object backend, float[] data, long[] shape) throws Exception {
        Class<?> backendClass = backend.getClass();
        // Will call: backend.create(Shape.of(shape), DType.F32, data)
        // For now, placeholder until Aljabr integration
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendZeros(Object backend, long[] shape) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendOnes(Object backend, long[] shape) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendRandn(Object backend, long[] shape) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendRand(Object backend, double low, double high, long[] shape) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeShape(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static boolean invokeRequiresGrad(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static void invokeSetRequiresGrad(Object tensor, boolean requiresGrad) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeGrad(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static float[] invokeGetData(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeBackendNeg(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendAbs(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendExp(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendLog(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendSqrt(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendClamp(Object backend, Object tensor, float min, float max) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendAdd(Object backend, Object a, Object b) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendSub(Object backend, Object a, Object b) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMul(Object backend, Object a, Object b) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendDiv(Object backend, Object a, Object b) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMatmul(Object backend, Object a, Object b) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMulScalar(Object backend, Object tensor, float scalar) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendSum(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMean(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMax(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeBackendMin(Object backend, Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr backend not yet integrated");
    }

    private static Object invokeReshape(Object tensor, long[] newShape) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeSqueeze(Object tensor, int dim) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeUnsqueeze(Object tensor, int dim) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeTranspose(Object tensor, int dim0, int dim1) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokePermute(Object tensor, int[] dims) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static void invokeBackward(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static void invokeBackwardWithGrad(Object tensor, Object gradOutput) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static void invokeZeroGrad(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }

    private static Object invokeDetach(Object tensor) throws Exception {
        throw new UnsupportedOperationException("Aljabr tensor not yet integrated");
    }
}
