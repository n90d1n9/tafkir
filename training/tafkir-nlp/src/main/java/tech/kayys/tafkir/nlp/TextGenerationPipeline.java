package tech.kayys.tafkir.nlp;

import tech.kayys.aljabr.sdk.api.AljabrSdk;
import tech.kayys.aljabr.sdk.api.AljabrSdkProvider;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

/**
 * Text generation pipeline backed by the Aljabr inference engine.
 *
 * <p>
 * Generates text auto-regressively from a prompt using temperature-based
 * sampling with optional top-p and top-k filtering. The pipeline resolves a
 * {@link AljabrSdk} instance at construction time via
 * {@link java.util.ServiceLoader}.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var pipeline = new TextGenerationPipeline("Qwen/Qwen2.5-0.5B");
 * String result = pipeline.generate("Tell me about Java", 200);
 * }</pre>
 *
 * @see PipelineFactory
 */
public class TextGenerationPipeline implements Pipeline<String, String> {

    private final String modelId;
    private final AljabrSdk sdk;
    private final PipelineConfig config;

    /**
     * Creates a pipeline with default configuration for the given model.
     *
     * @param modelId model identifier (e.g. {@code "Qwen/Qwen2.5-0.5B"})
     */
    public TextGenerationPipeline(String modelId) {
        this(PipelineConfig.builder("text-generation", modelId).build());
    }

    /**
     * Creates a pipeline with the given configuration.
     *
     * @param config pipeline configuration including model, sampling parameters,
     *               and device
     */
    public TextGenerationPipeline(PipelineConfig config) {
        this.config = config;
        this.modelId = config.modelId();
        this.sdk = resolveSdk();
    }

    /**
     * Generates text from the given prompt up to {@code maxTokens} tokens.
     *
     * @param prompt    the input prompt
     * @param maxTokens maximum number of tokens to generate
     * @return the generated text
     * @throws PipelineException if inference fails
     */
    public String generate(String prompt, int maxTokens) {
        if (sdk == null) {
            throw new PipelineException("No AljabrSdk provider available on classpath");
        }
        try {
            InferenceRequest request = InferenceRequest.builder()
                    .model(modelId)
                    .prompt(prompt)
                    .maxTokens(maxTokens)
                    .temperature(config.temperature())
                    .topP(config.topP())
                    .build();

            InferenceResponse response = sdk.createCompletion(request);
            return response.getContent();
        } catch (Exception e) {
            throw new PipelineException("Text generation failed: " + e.getMessage(), e);
        }
    }

    /**
     * Processes the input prompt using {@link PipelineConfig#maxLength()} as the
     * token limit.
     *
     * @param input the prompt string
     * @return the generated text
     * @throws PipelineException if inference fails
     */
    @Override
    public String process(String input) {
        return generate(input, config.maxLength());
    }

    @Override
    public String task() {
        return "text-generation";
    }

    @Override
    public String model() {
        return modelId;
    }

    /** Resolves a {@link AljabrSdk} via {@link java.util.ServiceLoader}. */
    private AljabrSdk resolveSdk() {
        try {
            return java.util.ServiceLoader.load(AljabrSdkProvider.class)
                    .findFirst()
                    .map(p -> p.create(null))
                    .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }
}
