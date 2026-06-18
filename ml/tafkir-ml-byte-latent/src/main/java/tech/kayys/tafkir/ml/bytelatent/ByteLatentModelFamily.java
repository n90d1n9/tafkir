package tech.kayys.tafkir.ml.bytelatent;

import java.util.List;

/**
 * Canonical metadata for Aljabr's byte-latent model family.
 */
public final class ByteLatentModelFamily {
    public static final String FAMILY_ID = "fast-byte-latent-transformer";
    public static final String DISPLAY_NAME = "Fast Byte Latent Transformer";
    public static final String PAPER_CITATION =
            "Julie Kallini et al., Fast Byte Latent Transformer, arXiv:2605.08044v1 (2026)";
    public static final String DOI = "10.48550/arXiv.2605.08044";

    private ByteLatentModelFamily() {
    }

    public static List<String> recommendedModuleIds() {
        return List.of(
                "ml:tafkir-ml-language-core",
                "ml:tafkir-ml-byte-latent",
                "ml:tafkir-ml-byte-io",
                "runner:aljabr-runner-byte-latent",
                "trainer:aljabr-trainer-byte-latent");
    }
}
