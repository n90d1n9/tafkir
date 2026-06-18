package tech.kayys.tafkir.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.tafkir.sdk.api.TafkirClient;

import java.util.List;
import java.util.Objects;

/**
 * Tafkir implementation of LangChain4j ChatLanguageModel.
 */
public class TafkirChatModel implements ChatLanguageModel {

    private final TafkirClient client;
    private final String model;
    private final float temperature;
    private final int maxTokens;

    private TafkirChatModel(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client is required");
        this.model = builder.model;
        this.temperature = builder.temperature;
        this.maxTokens = builder.maxTokens;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        // Check if there are multimodal parts
        boolean isMultimodal = messages.stream()
                .anyMatch(m -> m instanceof UserMessage um && um.contents() != null && 
                        um.contents().stream().anyMatch(c -> !(c instanceof TextContent)));

        if (isMultimodal) {
            var parts = TafkirMessageMapper.toParts(messages);
            var result = tech.kayys.tafkir.ml.Tafkir.multimodal(model)
                    .parts(parts)
                    .generate();
            return Response.from(AiMessage.from(result.text()));
        }

        // Standard text generation
        String prompt = TafkirMessageMapper.toPrompt(messages);

        var result = client.generate(prompt);

        return Response.from(AiMessage.from(result.text()));
    }

    public static class Builder {
        private TafkirClient client;
        private String model;
        private float temperature = 0.7f;
        private int maxTokens = 512;

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

        public Builder temperature(double temperature) {
            this.temperature = (float) temperature;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public TafkirChatModel build() {
            return new TafkirChatModel(this);
        }
    }
}
