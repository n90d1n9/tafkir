package tech.kayys.aljabr.safetensor.loader;

import java.util.Map;

/**
 * Stub — parsed header of a single safetensor file, containing
 * tensor metadata and optional file-level metadata map.
 */
public final class SafetensorHeader {

    private final long dataOffset;
    private final Map<String, SafetensorTensorInfo> tensors;
    private final Map<String, String> fileMetadata;

    private SafetensorHeader(long dataOffset,
                              Map<String, SafetensorTensorInfo> tensors,
                              Map<String, String> fileMetadata) {
        this.dataOffset = dataOffset;
        this.tensors = tensors;
        this.fileMetadata = fileMetadata;
    }

    public static SafetensorHeader of(long dataOffset,
                                      Map<String, SafetensorTensorInfo> tensors,
                                      Map<String, String> fileMetadata) {
        return new SafetensorHeader(dataOffset, tensors, fileMetadata);
    }

    public long dataOffset() { return dataOffset; }

    public Map<String, SafetensorTensorInfo> tensors() { return tensors; }

    public Map<String, String> fileMetadata() { return fileMetadata; }
}
