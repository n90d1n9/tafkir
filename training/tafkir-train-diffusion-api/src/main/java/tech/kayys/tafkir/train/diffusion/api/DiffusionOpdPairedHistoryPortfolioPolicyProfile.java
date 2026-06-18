package tech.kayys.tafkir.train.diffusion.api;

/**
 * Typed paired portfolio evaluation under one status policy profile.
 */
public record DiffusionOpdPairedHistoryPortfolioPolicyProfile(
        String policyName,
        DiffusionOpdRoundHistoryStatusPolicy policy,
        DiffusionOpdPairedHistoryPortfolioExecutiveSummary executiveSummary,
        DiffusionOpdPairedHistoryPortfolioRemediationSummary remediationSummary) {
}
