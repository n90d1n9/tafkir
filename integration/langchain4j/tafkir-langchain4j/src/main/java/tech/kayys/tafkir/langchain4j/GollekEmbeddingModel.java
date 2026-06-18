package tech.kayys.tafkir.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.tafkir.sdk.api.TafkirClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Tafkir implementation of LangChain4j EmbeddingModel.
 */
public class TafkirEmbeddingModel implements EmbeddingModel {

    private final TafkirClient client;
    private final String model;

    private TafkirEmbeddingModel(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client is required");
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : textSegments) {
            float[] vector = client.embed(segment.text());
            embeddings.add(Embedding.from(vector));
        }
        return Response.from(embeddings);
    }

    public static class Builder {
        private TafkirClient client;
        private String model;

        public Builder client(TafkirClient client) {
            this.client = client;
            return this;
        }

        public Builder endpoint(String endpoint) {
            this.client = TafkirClient.builder().endpoint(endpoint).build();
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public TafkirEmbeddingModel build() {
            return new TafkirEmbeddingModel(this);
        }
    }
}
