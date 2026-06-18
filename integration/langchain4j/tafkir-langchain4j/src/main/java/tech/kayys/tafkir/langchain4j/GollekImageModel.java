package tech.kayys.tafkir.langchain4j;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.tafkir.ml.Tafkir;
import tech.kayys.tafkir.ml.multimodal.MultimodalResult;

import java.util.Collections;
import java.util.List;
import java.util.Base64;

/**
 * Tafkir implementation of LangChain4j ImageModel for text-to-image generation.
 */
public class TafkirImageModel implements ImageModel {

    private final String model;
    private final String quality;
    private final String size;

    private TafkirImageModel(Builder builder) {
        this.model = builder.model;
        this.quality = builder.quality;
        this.size = builder.size;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<Image> generate(String prompt) {
        // Use the new Tafkir fluent API for vision/image generation
        MultimodalResult result = Tafkir.vision(model)
                .prompt(prompt)
                .generate();

        // Check if we have binary data (image)
        if (result.hasBinary()) {
            byte[] data = result.binaryData();
            String base64 = Base64.getEncoder().encodeToString(data);
            
            // For now, returning as base64 data. 
            // LangChain4j Image can also take a URL or Path.
            Image image = Image.builder()
                    .base64Data(base64)
                    .build();
            
            return Response.from(image);
        }

        throw new RuntimeException("Tafkir failed to generate an image for the given prompt. " +
                "Response text: " + result.text());
    }

    @Override
    public Response<List<Image>> generate(String prompt, int n) {
        // Simple implementation: generate n images sequentially
        // In a production backend, this might be a single batch request
        return Response.from(Collections.singletonList(generate(prompt).content()));
    }

    public static class Builder {
        private String model = "stable-diffusion-xl";
        private String quality = "standard";
        private String size = "1024x1024";

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder quality(String quality) {
            this.quality = quality;
            return this;
        }

        public Builder size(String size) {
            this.size = size;
            return this;
        }

        public TafkirImageModel build() {
            return new TafkirImageModel(this);
        }
    }
}
