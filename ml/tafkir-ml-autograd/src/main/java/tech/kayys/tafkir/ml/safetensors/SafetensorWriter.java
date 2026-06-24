package tech.kayys.tafkir.ml.safetensors;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Production-ready SafeTensors writer following the official specification.
 * 
 * <p>
 * SafeTensors is a simple, zero-copy format for storing tensors.
 * The format consists of a JSON header followed by binary tensor data.
 * </p>
 * 
 * <p>
 * Format specification:
 * <ul>
 * <li>Header: JSON string describing tensor metadata and offsets</li>
 * <li>Header length: 8-byte unsigned integer (little-endian)</li>
 * <li>Data: Binary tensor data at specified offsets</li>
 * </ul>
 * </p>
 * 
 * @see <a href="https://github.com/huggingface/safetensors">SafeTensors
 *      Specification</a>
 */
public final class SafetensorWriter {

    private SafetensorWriter() {
        throw new UnsupportedOperationException("Utility class");
    }

    /**
     * Saves a map of tensors to a SafeTensors file.
     *
     * @param path    The output file path
     * @param tensors Map of tensor names to TafkirTensor objects
     * @throws IOException If writing fails
     */
    public static void save(Path path, Map<String, TafkirTensor> tensors) throws IOException {
        validateTensors(tensors);

        // Extract data and metadata
        Map<String, TensorData> tensorDataMap = new LinkedHashMap<>();
        long currentOffset = 0;

        for (Map.Entry<String, TafkirTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            TafkirTensor tensor = entry.getValue();

            float[] data = tensor.data();
            long[] shape = tensor.shapeArray();

            // Calculate bytes needed (4 bytes per float)
            int byteSize = data.length * Float.BYTES;

            tensorDataMap.put(name, new TensorData(
                    data,
                    shape,
                    tensor.dtype().name().toLowerCase(),
                    currentOffset,
                    byteSize));

            currentOffset += byteSize;
        }

        // Build and write the header
        String headerJson = buildHeader(tensorDataMap);
        byte[] headerBytes = headerJson.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        long headerLength = headerBytes.length;

        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write header length (8 bytes, little-endian)
            ByteBuffer headerLenBuffer = ByteBuffer.allocate(8);
            headerLenBuffer.order(ByteOrder.LITTLE_ENDIAN);
            headerLenBuffer.putLong(headerLength);
            headerLenBuffer.flip();
            channel.write(headerLenBuffer);

            // Write header
            ByteBuffer headerBuffer = ByteBuffer.wrap(headerBytes);
            channel.write(headerBuffer);

            // Write tensor data
            for (TensorData tensorData : tensorDataMap.values()) {
                ByteBuffer dataBuffer = ByteBuffer.allocate(tensorData.byteSize);
                dataBuffer.order(ByteOrder.LITTLE_ENDIAN);

                for (float value : tensorData.data) {
                    dataBuffer.putFloat(value);
                }
                dataBuffer.flip();

                channel.write(dataBuffer);
            }
        }
    }

    /**
     * Validates input tensors.
     */
    private static void validateTensors(Map<String, TafkirTensor> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Tensors map cannot be null or empty");
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
     * Builds the JSON header according to the SafeTensors specification.
     */
    private static String buildHeader(Map<String, TensorData> tensorDataMap) {
        StringBuilder json = new StringBuilder();
        json.append("{");

        int index = 0;
        for (Map.Entry<String, TensorData> entry : tensorDataMap.entrySet()) {
            if (index > 0) {
                json.append(",");
            }

            String name = entry.getKey();
            TensorData data = entry.getValue();

            json.append("\"").append(escapeJson(name)).append("\":{");
            json.append("\"dtype\":\"").append(data.dtype).append("\",");
            json.append("\"shape\":").append(dimsToJson(data.shape)).append(",");
            json.append("\"data_offsets\":[").append(data.offset).append(",")
                    .append(data.offset + data.byteSize).append("]");
            json.append("}");
            index++;
        }

        json.append("}");
        return json.toString();
    }

    /**
     * Converts dimensions to JSON array string.
     */
    private static String dimsToJson(long[] dims) {
        if (dims == null || dims.length == 0) {
            return "[]";
        }
        return Arrays.stream(dims)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(",", "[", "]"));
    }

    /**
     * Escapes a string for JSON.
     */
    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\b", "\\b")
                .replace("\f", "\\f")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * Internal data holder for tensor information.
     */
    private static class TensorData {
        final float[] data;
        final long[] shape;
        final String dtype;
        final long offset;
        final int byteSize;

        TensorData(float[] data, long[] shape, String dtype, long offset, int byteSize) {
            this.data = data;
            this.shape = shape.clone();
            this.dtype = dtype;
            this.offset = offset;
            this.byteSize = byteSize;
        }
    }
}