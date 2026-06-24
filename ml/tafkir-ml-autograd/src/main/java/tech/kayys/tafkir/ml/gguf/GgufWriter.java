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
 * Production-ready GGUF (GGML Universal Format) writer.
 * 
 * <p>
 * GGUF is a binary format for storing machine learning models, used by
 * llama.cpp and other GGML-based projects. It supports multiple data types,
 * metadata, and efficient tensor storage.
 * </p>
 * 
 * <p>
 * Format specification:
 * <ul>
 * <li>Magic number: "GGUF" (0x46554747)</li>
 * <li>Version: 3 (uint32_t)</li>
 * <li>Tensor count: uint64_t</li>
 * <li>Metadata key-value pairs: variable length</li>
 * <li>Tensor data: aligned to 32 bytes</li>
 * </ul>
 * </p>
 * 
 * @see <a href=
 *      "https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF
 *      Specification</a>
 */
public final class GgufWriter {

    private static final int GGUF_MAGIC = 0x46554747; // "GGUF"
    private static final int GGUF_VERSION = 3;
    private static final int ALIGNMENT = 32;

    private GgufWriter() {
        throw new UnsupportedOperationException("Utility class");
    }

    /**
     * Saves a map of tensors to a GGUF file.
     *
     * @param path     The output file path
     * @param tensors  Map of tensor names to TafkirTensor objects
     * @param metadata Metadata key-value pairs
     * @throws IOException If writing fails
     */
    public static void save(Path path, Map<String, TafkirTensor> tensors,
            Map<String, GgufMetaValue> metadata) throws IOException {
        validateInputs(tensors, metadata);

        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write header
            writeHeader(channel, tensors.size());

            // Write metadata
            writeMetadata(channel, metadata);

            // Write alignment padding after metadata
            alignTo32(channel);

            // Write tensor info and data
            writeTensors(channel, tensors);
        }
    }

    /**
     * Validates inputs.
     */
    private static void validateInputs(Map<String, TafkirTensor> tensors,
            Map<String, GgufMetaValue> metadata) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Tensors map cannot be null or empty");
        }
        if (metadata == null) {
            throw new IllegalArgumentException("Metadata cannot be null");
        }

        for (Map.Entry<String, TafkirTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            TafkirTensor tensor = entry.getValue();

            if (tensor == null) {
                throw new IllegalArgumentException("Tensor '" + name + "' is null");
            }

            float[] data = tensor.data();
            if (data == null || data.length == 0) {
                throw new IllegalArgumentException("Tensor '" + name + "' has no data");
            }
        }
    }

    /**
     * Writes the GGUF header.
     */
    private static void writeHeader(FileChannel channel, int tensorCount) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(20); // 4 + 4 + 8 + 4 = 20 bytes
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        buffer.putInt(GGUF_MAGIC);
        buffer.putInt(GGUF_VERSION);
        buffer.putLong(tensorCount); // tensor_count
        buffer.putInt(0); // metadata_kv_count (will be handled separately)

        buffer.flip();
        channel.write(buffer);
    }

    /**
     * Writes metadata key-value pairs.
     */
    private static void writeMetadata(FileChannel channel, Map<String, GgufMetaValue> metadata)
            throws IOException {
        // Write metadata count
        ByteBuffer countBuffer = ByteBuffer.allocate(8);
        countBuffer.order(ByteOrder.LITTLE_ENDIAN);
        countBuffer.putLong(metadata.size());
        countBuffer.flip();
        channel.write(countBuffer);

        // Write each metadata entry
        for (Map.Entry<String, GgufMetaValue> entry : metadata.entrySet()) {
            writeMetadataEntry(channel, entry.getKey(), entry.getValue());
        }
    }

    /**
     * Writes a single metadata entry.
     */
    private static void writeMetadataEntry(FileChannel channel, String key, GgufMetaValue value)
            throws IOException {
        // Write key
        writeString(channel, key);

        // Write value type
        ByteBuffer typeBuffer = ByteBuffer.allocate(4);
        typeBuffer.order(ByteOrder.LITTLE_ENDIAN);
        typeBuffer.putInt(value.getType().getCode());
        typeBuffer.flip();
        channel.write(typeBuffer);

        // Write value based on type
        switch (value.getType()) {
            case UINT8:
                writeUint8(channel, value.getUint8Value());
                break;
            case INT8:
                writeInt8(channel, value.getInt8Value());
                break;
            case UINT16:
                writeUint16(channel, value.getUint16Value());
                break;
            case INT16:
                writeInt16(channel, value.getInt16Value());
                break;
            case UINT32:
                writeUint32(channel, value.getUint32Value());
                break;
            case INT32:
                writeInt32(channel, value.getInt32Value());
                break;
            case FLOAT32:
                writeFloat32(channel, value.getFloat32Value());
                break;
            case UINT64:
                writeUint64(channel, value.getUint64Value());
                break;
            case INT64:
                writeInt64(channel, value.getInt64Value());
                break;
            case FLOAT64:
                writeFloat64(channel, value.getFloat64Value());
                break;
            case BOOL:
                writeBool(channel, value.getBoolValue());
                break;
            case STRING:
                writeString(channel, value.getStringValue());
                break;
            case ARRAY:
                writeArray(channel, value.getArrayValue());
                break;
            default:
                throw new UnsupportedOperationException("Unsupported metadata type: " + value.getType());
        }
    }

    /**
     * Writes a string with length prefix.
     */
    private static void writeString(FileChannel channel, String s) throws IOException {
        byte[] bytes = s.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(8 + bytes.length);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(bytes.length);
        buffer.put(bytes);
        buffer.flip();
        channel.write(buffer);
    }

    /**
     * Writes tensor information and data.
     */
    private static void writeTensors(FileChannel channel, Map<String, TafkirTensor> tensors)
            throws IOException {
        for (Map.Entry<String, TafkirTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            TafkirTensor tensor = entry.getValue();
            float[] data = tensor.data();
            long[] shape = tensor.shapeArray();

            // Write tensor name
            writeString(channel, name);

            // Write tensor shape (n_dims followed by dims)
            ByteBuffer shapeBuffer = ByteBuffer.allocate(8 + shape.length * 8);
            shapeBuffer.order(ByteOrder.LITTLE_ENDIAN);
            shapeBuffer.putLong(shape.length);
            for (long dim : shape) {
                shapeBuffer.putLong(dim);
            }
            shapeBuffer.flip();
            channel.write(shapeBuffer);

            // Write tensor data type (we'll use FLOAT32 for now)
            ByteBuffer dtypeBuffer = ByteBuffer.allocate(4);
            dtypeBuffer.order(ByteOrder.LITTLE_ENDIAN);
            dtypeBuffer.putInt(GgufDType.F32.getCode());
            dtypeBuffer.flip();
            channel.write(dtypeBuffer);

            // Write tensor offset (will be updated later)
            long offsetPosition = channel.position();
            ByteBuffer offsetBuffer = ByteBuffer.allocate(8);
            offsetBuffer.order(ByteOrder.LITTLE_ENDIAN);
            offsetBuffer.putLong(0); // Placeholder
            offsetBuffer.flip();
            channel.write(offsetBuffer);

            // Align to 32 bytes before writing data
            alignTo32(channel);

            // Write tensor data
            long dataStartPosition = channel.position();

            // Convert float[] to bytes and write
            ByteBuffer dataBuffer = ByteBuffer.allocate(data.length * 4);
            dataBuffer.order(ByteOrder.LITTLE_ENDIAN);
            for (float f : data) {
                dataBuffer.putFloat(f);
            }
            dataBuffer.flip();
            channel.write(dataBuffer);

            // Go back and update the offset
            long currentPosition = channel.position();
            channel.position(offsetPosition);
            ByteBuffer updatedOffset = ByteBuffer.allocate(8);
            updatedOffset.order(ByteOrder.LITTLE_ENDIAN);
            updatedOffset.putLong(dataStartPosition);
            updatedOffset.flip();
            channel.write(updatedOffset);

            // Return to end of file
            channel.position(currentPosition);
        }
    }

    /**
     * Aligns the file position to 32 bytes.
     */
    private static void alignTo32(FileChannel channel) throws IOException {
        long position = channel.position();
        long remainder = position % ALIGNMENT;
        if (remainder != 0) {
            int padding = (int) (ALIGNMENT - remainder);
            ByteBuffer paddingBuffer = ByteBuffer.allocate(padding);
            channel.write(paddingBuffer);
        }
    }

    // ── Type-specific write methods ──

    private static void writeUint8(FileChannel channel, byte value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1);
        buffer.put(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeInt8(FileChannel channel, byte value) throws IOException {
        writeUint8(channel, value);
    }

    private static void writeUint16(FileChannel channel, short value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putShort(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeInt16(FileChannel channel, short value) throws IOException {
        writeUint16(channel, value);
    }

    private static void writeUint32(FileChannel channel, int value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeInt32(FileChannel channel, int value) throws IOException {
        writeUint32(channel, value);
    }

    private static void writeFloat32(FileChannel channel, float value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putFloat(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeUint64(FileChannel channel, long value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeInt64(FileChannel channel, long value) throws IOException {
        writeUint64(channel, value);
    }

    private static void writeFloat64(FileChannel channel, double value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putDouble(value);
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeBool(FileChannel channel, boolean value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1);
        buffer.put((byte) (value ? 1 : 0));
        buffer.flip();
        channel.write(buffer);
    }

    private static void writeArray(FileChannel channel, List<GgufMetaValue> values) throws IOException {
        if (values == null || values.isEmpty()) {
            ByteBuffer buffer = ByteBuffer.allocate(8);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.putLong(0);
            buffer.flip();
            channel.write(buffer);
            return;
        }

        // Write array length
        ByteBuffer lenBuffer = ByteBuffer.allocate(8);
        lenBuffer.order(ByteOrder.LITTLE_ENDIAN);
        lenBuffer.putLong(values.size());
        lenBuffer.flip();
        channel.write(lenBuffer);

        // Write each array element
        for (GgufMetaValue value : values) {
            writeMetadataEntry(channel, "", value); // Empty key for array elements
        }
    }
}