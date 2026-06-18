package tech.kayys.tafkir.langchain4j;

import dev.langchain4j.model.chat.ChatLanguageModel;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.tafkir.sdk.api.TafkirClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

public class TafkirChatModelTest {

    @Test
    void should_generate_response() {
        // Given
        TafkirClient mockClient = Mockito.mock(TafkirClient.class);
        TafkirClient.GenerationResult mockResult = new TafkirClient.GenerationResult(
                "Hello from Tafkir!", 4, 1, 100);
        
        when(mockClient.generate(anyString())).thenReturn(mockResult);

        ChatLanguageModel model = TafkirChatModel.builder()
                .client(mockClient)
                .build();

        // When
        String response = model.generate("Hi");

        // Then
        assertThat(response).isEqualTo("Hello from Tafkir!");
    }
}
