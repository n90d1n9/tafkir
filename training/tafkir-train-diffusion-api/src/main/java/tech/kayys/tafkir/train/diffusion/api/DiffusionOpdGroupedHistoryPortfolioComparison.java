package tech.kayys.tafkir.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped portfolio comparison across two runs.
 */
public record DiffusionOpdGroupedHistoryPortfolioComparison(
        String field,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> entries) {
}
