package tech.kayys.tafkir.ml.bytelatent;

/**
 * Decodes latent codes back to byte-native sequences.
 */
public interface ByteLatentDecoder {
    ByteSequenceBatch decode(ByteLatentState state);
}
