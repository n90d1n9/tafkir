package tech.kayys.tafkir.ml.bytelatent;

/**
 * Encodes byte-native sequences into latent codes.
 */
public interface ByteLatentEncoder {
    ByteLatentState encode(ByteSequenceBatch batch);
}
