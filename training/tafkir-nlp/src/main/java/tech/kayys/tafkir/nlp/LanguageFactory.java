package tech.kayys.tafkir.nlp;

import tech.kayys.aljabr.tokenizer.TokenizerFactory;
import tech.kayys.aljabr.tokenizer.spi.EncodeOptions;
import tech.kayys.aljabr.tokenizer.spi.ModelConfig;
import tech.kayys.aljabr.tokenizer.spi.TokenizerType;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Factory for creating Language pipelines.
 */
public class LanguageFactory {

    public static Language load(String modelName) {
        String langCode = modelName.split("_")[0];
        Path tokenizerPath = Paths.get("models", modelName, "tokenizer.json");

        Language.Tokenizer tokenizer;
        try {
            var modelConfig = new ModelConfig(TokenizerType.BPE, tokenizerPath);
            var aljabrTokenizer = TokenizerFactory.create(modelConfig);

            tokenizer = text -> {
                Doc doc = new Doc(text);
                long[] ids = aljabrTokenizer.encode(text, EncodeOptions.defaultOptions());

                // Simplified mapping for demonstration
                String[] words = text.split("\\s+");
                for (int i = 0; i < words.length; i++) {
                    doc.addToken(Token.builder()
                            .doc(doc)
                            .i(i)
                            .text(words[i])
                            .isAlpha(words[i].matches("[a-zA-Z]+"))
                            .build());
                }
                return doc;
            };
        } catch (Exception e) {
            tokenizer = text -> {
                Doc doc = new Doc(text);
                String[] words = text.split("\\s+");
                for (int i = 0; i < words.length; i++) {
                    doc.addToken(Token.builder().doc(doc).i(i).text(words[i]).build());
                }
                return doc;
            };
        }

        Language nlp = new Language(langCode, tokenizer);

        try {
            EmbeddingPipeline ep = new EmbeddingPipeline(modelName);
            nlp.addPipe(new EmbeddingProcessor(ep));
        } catch (Exception e) {
            // No embedding model available
        }

        return nlp;
    }
}
