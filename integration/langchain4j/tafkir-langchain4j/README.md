# Tafkir SDK :: LangChain4j Integration
 
 `tafkir-langchain4j` acts as the bridge connecting the powerful Tafkir LLM runtime natively into the Java LangChain4j ecosystem.
 
 ## Capabilities
 - **ChatLanguageModel**: Map `TafkirSdk` pipelines to LangChain4j Chat representations (`TafkirChatModel`).
 - **StreamingChatLanguageModel**: Asynchronous callback streams bound directly to `tafkir-engine` token emit hooks (`TafkirStreamingChatModel`).
 - **Embeddings**: Native bridge from the Tafkir Vision/Text embeddings pipelines.
 
 ## Architecture Mapping
 This sub-module operates inside the library tier (`tafkir-sdk-ml-parent`) and interfaces entirely with `tafkir-sdk-ml`, translating generic string buffers to `AiMessage`, `UserMessage`, and `SystemMessage` constructs automatically.
 
 ```java
 import dev.langchain4j.model.chat.ChatLanguageModel;
 import tech.kayys.tafkir.ml.Tafkir;
 import tech.kayys.tafkir.langchain4j.TafkirChatModel;
 
 Tafkir gnk = Tafkir.builder().model("Llama-3").build();
 ChatLanguageModel model = new TafkirChatModel(gnk);
 
 String answer = model.generate("Hi, how are you?");
 ```
