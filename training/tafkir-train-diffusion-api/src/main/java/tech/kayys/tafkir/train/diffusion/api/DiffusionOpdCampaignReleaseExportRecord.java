package tech.kayys.tafkir.train.diffusion.api;

/**
 * Typed release export record for one release-handoff artifact.
 */
public record DiffusionOpdCampaignReleaseExportRecord(
        String artifactType,
        String destination,
        String deliveryStatus,
        String referenceKey) {
}
