package tech.kayys.aljabr.hub;

import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * @deprecated Use {@link tech.kayys.tafkir.ml.hub.ModelHub} or
 *             {@link tech.kayys.tafkir.ml.Aljabr.Hub} instead.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
@SuppressWarnings("removal")
public final class ModelHub {

    private ModelHub() {
    }

    public static void loadInto(NNModule model, String modelId) throws IOException {
        tech.kayys.tafkir.ml.hub.ModelHub.loadInto(model, modelId);
    }

    public static void loadInto(NNModule model, String modelId, HubConfig config) throws IOException {
        tech.kayys.tafkir.ml.hub.ModelHub.loadInto(model, modelId, config.toCanonical());
    }

    public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
        return tech.kayys.tafkir.ml.hub.ModelHub.loadWeights(modelId);
    }

    public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) throws IOException {
        return tech.kayys.tafkir.ml.hub.ModelHub.loadWeights(modelId, config.toCanonical());
    }

    public static void setCacheDir(Path dir) {
        // Compatibility no-op; prefer HubConfig.builder().cacheDir(...)
    }
}
