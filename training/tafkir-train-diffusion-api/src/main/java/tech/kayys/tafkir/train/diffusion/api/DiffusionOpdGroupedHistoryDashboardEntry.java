package tech.kayys.tafkir.train.diffusion.api;

/**
 * Typed grouped dashboard entry with summary and richer slice analysis.
 */
public record DiffusionOpdGroupedHistoryDashboardEntry(
        String field,
        String value,
        DiffusionOpdRoundHistorySummary summary,
        DiffusionOpdRoundHistorySnapshot snapshot,
        DiffusionOpdRoundHistoryTimeline timeline) {
}
