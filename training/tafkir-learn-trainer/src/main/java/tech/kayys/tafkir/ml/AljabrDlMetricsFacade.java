package tech.kayys.tafkir.ml;

/**
 * Public metric factory facade inherited by {@link Aljabr.DL}.
 *
 * <p>The methods intentionally mirror the historical {@code Aljabr.DL.*Metric}
 * API so callers keep the same import style while each metric family evolves in
 * its own focused facade.
 */
public class AljabrDlMetricsFacade extends AljabrDlLanguageModelingMetricsFacade {
    protected AljabrDlMetricsFacade() {
    }
}
