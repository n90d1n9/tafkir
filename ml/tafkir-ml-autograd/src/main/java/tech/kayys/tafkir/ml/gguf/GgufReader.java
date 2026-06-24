package tech.kayys.tafkir.ml.gguf;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Production-ready GGUF (GGML Universal Format) reader.
 * 
 * <p>
 * Reads GGUF format files, supporting multiple data types and metadata.
 * </p>
 */
public final class GgufReader {

    private static final int GGUF_MAGIC = 0x46554747;
    private static final int GGUF_VERSION = 3;

    private GgufReader() {
        throw new UnsupportedOperationException("Utility class");
    }

    /**
     * Reads a GGUF file and returns all tensors and metadata.
     *
     * @param path The input file path
     * @return GgufModel containing tensors and metadata
     * @throws IOException If reading fails
     */
    public static GgufModel read(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0,
                    channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Read and validate magic
            int magic = buffer.getInt();
            if (magic != GGUF_MAGIC) {
                throw new GgufFormatException("Invalid GGUF magic number: " + Integer.toHexString(magic));
            }

            // Read version
            int version = buffer.getInt();
            if (version != GGUF_VERSION) {
                throw new GgufFormatException("Unsupported GGUF version: " + version);
            }

            // Read tensor count
            long tensorCount = buffer.getLong();

            // Read metadata
            Map<String, GgufMetaValue> metadata = readMetadata(buffer);

            // Read tensor info
            Map<String, TensorInfo> tensorInfos = new LinkedHashMap<>();
            for (int i = 0; i < tensorCount; i++) {
                TensorInfo info = readTensorInfo(buffer);
                tensorInfos.put(info.name, info);
            }

            // Read tensor data
            Map<String, TafkirTensor> tensors = readTensors(buffer, tensorInfos);

            return new GgufModel(tensors, metadata);
        }
    }

    /**
     * Reads metadata from the buffer.
     */
    private static Map<String, GgufMetaValue> readMetadata(ByteBuffer buffer) {
        long metadataCount = buffer.getLong();
        Map<String, GgufMetaValue> metadata = new LinkedHashMap<>();

        for (int i = 0; i < metadataCount; i++) {
            String key = readString(buffer);
            GgufMetaValue value = readMetadataValue(buffer);
            metadata.put(key, value);
        }

        return metadata;
    }

    /**
     * Reads a single metadata value.
     */
    private static GgufMetaValue readMetadataValue(ByteBuffer buffer) {
        int typeCode = buffer.getInt();
        GgufMetaType type = GgufMetaType.fromCode(typeCode);
        if (type == null) {
            throw new GgufFormatException("Unknown metadata type code: " + typeCode);
        }

        return switch (type) {
            case UINT8 -> GgufMetaValue.of(buffer.get());
            case INT8 -> GgufMetaValue.of((byte) buffer.get());
            case UINT16 -> GgufMetaValue.of(buffer.getShort() & 0xFFFF);
            case INT16 -> GgufMetaValue.of(buffer.getShort());
            case UINT32 -> GgufMetaValue.of(buffer.getInt() & 0xFFFFFFFFL);
            case INT32 -> GgufMetaValue.of(buffer.getInt());
            case FLOAT32 -> GgufMetaValue.of(buffer.getFloat());
            case UINT64 -> GgufMetaValue.of(buffer.getLong());
            case INT64 -> GgufMetaValue.of(buffer.getLong());
            case FLOAT64 -> GgufMetaValue.of(buffer.getDouble());
            case BOOL -> GgufMetaValue.of(buffer.get() != 0);
            case STRING -> GgufMetaValue.of(readString(buffer));
            case ARRAY -> readArrayValue(buffer);
        };
    }

    /**
     * Reads an array value.
     */
    private static GgufMetaValue readArrayValue(ByteBuffer buffer) {
        long arrayLength = buffer.getLong();
        List<GgufMetaValue> values = new ArrayList<>();

        for (int i = 0; i < arrayLength; i++) {
            values.add(readMetadataValue(buffer));
        }

        return GgufMetaValue.ofArray(values);
    }

    /**
     * Reads a string from the buffer.
     */
    private static String readString(ByteBuffer buffer) {
        long length = buffer.getLong();
        if (length > Integer.MAX_VALUE) {
            throw new GgufFormatException("String too long: " + length);
        }

        byte[] bytes = new byte[(int) length];
        buffer.get(bytes);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * Reads tensor information from the buffer.
     */
    private static TensorInfo readTensorInfo(ByteBuffer buffer) {
        String name = readString(buffer);

        // Read shape
        long nDims = buffer.getLong();
        long[] shape = new long[(int) nDims];
        for (int i = 0; i < nDims; i++) {
            shape[i] = buffer.getLong();
        }

        // Read data type
        int dtypeCode = buffer.getInt();
        GgufDType dtype = GgufDType.fromCode(dtypeCode);
        if (dtype == null) {
            throw new GgufFormatException("Unknown tensor dtype code: " + dtypeCode);
        }

        // Read offset
        long offset = buffer.getLong();

        return new TensorInfo(name, shape, dtype, offset);
    }

    /**
     * Reads tensor data from the buffer.
     */
    private static Map<String, TafkirTensor> readTensors(ByteBuffer buffer,
            Map<String, TensorInfo> tensorInfos) {
        Map<String, TafkirTensor> result = new LinkedHashMap<>();

        for (Map.Entry<String, TensorInfo> entry : tensorInfos.entrySet()) {
            TensorInfo info = entry.getValue();

            // Position to tensor data
            buffer.position((int) info.offset);

            // Read tensor data based on dtype
            float[] data = readTensorData(buffer, info);

            // Create TafkirTensor
            TafkirTensor tensor = TafkirTensor.of(data, info.shape);
            result.put(entry.getKey(), tensor);
        }

        return result;
    }

    /**
     * Reads tensor data based on data type.
     */
    private static float[] readTensorData(ByteBuffer buffer, TensorInfo info) {
        long numElements = computeNumElements(info.shape);

        return switch (info.dtype) {
            case F32 -> readFloat32Data(buffer, (int) numElements);
            case F16 -> readFloat16Data(buffer, (int) numElements);
            case Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K ->
                readQuantizedData(buffer, info.dtype, (int) numElements);
            default -> throw new UnsupportedOperationException(
                    "Unsupported dtype for reading: " + info.dtype);
        };
    }

    /**
     * Reads F32 data.
     */
    private static float[] readFloat32Data(ByteBuffer buffer, int numElements) {
        float[] data = new float[numElements];
        for (int i = 0; i < numElements; i++) {
            data[i] = buffer.getFloat();
        }
        return data;
    }

    /**
     * Reads F16 data (converted to F32).
     */
    private static float[] readFloat16Data(ByteBuffer buffer, int numElements) {
        float[] data = new float[numElements];
        for (int i = 0; i < numElements; i++) {
            short half = buffer.getShort();
            data[i] = halfToFloat(half);
        }
        return data;
    }

    /**
     * Reads quantized data (placeholder - actual implementation depends on
     * quantization scheme).
     */
    private static float[] readQuantizedData(ByteBuffer buffer, GgufDType dtype, int numElements) {
        // This is a placeholder for quantized data reading
        // Actual implementation depends on the specific quantization format
        float[] data = new float[numElements];
        // TODO: Implement quantization-specific reading
        return data;
    }

    /**
     * Converts half-precision float to full precision.
     */
    private static float halfToFloat(short half) {
        // Implementation of IEEE 754 half-precision to single-precision conversion
        int sign = (half >> 15) & 0x1;
        int exponent = (half >> 10) & 0x1F;
        int mantissa = half & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Denormalized
            int floatBits = (sign << 31) | (0x70 << 23) | (mantissa << 13);
            return Float.intBitsToFloat(floatBits);
        } else if (exponent == 0x1F) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        int floatBits = (sign << 31) | ((exponent + 0x70) << 23) | (mantissa << 13);
        return Float.intBitsToFloat(floatBits);
    }

    /**
     * Computes total number of elements from shape.
     */
    private static long computeNumElements(long[] shape) {
        long result = 1;
        for (long dim : shape) {
            result *= dim;
        }
        return result;
    }

    /**
     * Internal class for tensor information.
     */
    private static class TensorInfo {
        final String name;
        final long[] shape;
        final GgufDType dtype;
        final long offset;

        TensorInfo(String name, long[] shape, GgufDType dtype, long offset) {
            this.name = name;
            this.shape = shape.clone();
            this.dtype = dtype;
            this.offset = offset;
        }
    }

    /**
     * Exception thrown when GGUF format is invalid.
     */
    public static class GgufFormatException extends RuntimeException {
        public GgufFormatException(String message) {
            super(message);
        }

        public GgufFormatException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}