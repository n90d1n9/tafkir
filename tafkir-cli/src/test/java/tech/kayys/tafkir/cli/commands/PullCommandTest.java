package tech.kayys.tafkir.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import jakarta.inject.Inject;

import tech.kayys.tafkir.sdk.core.TafkirSdk;
import tech.kayys.tafkir.sdk.model.ModelPullRequest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@QuarkusTest
public class PullCommandTest {

    @Inject
    PullCommand pullCommand;

    @InjectMock
    TafkirSdk sdk;

    @Test
    public void testPullCommandWithSimpleName() throws Exception {
        pullCommand.modelSpec = "llama3.2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(any(ModelPullRequest.class), any());
    }

    @Test
    public void testPullCommandHuggingFace() throws Exception {
        pullCommand.modelSpec = "hf:TheBloke/Llama-2";
        pullCommand.insecure = false;

        pullCommand.run();

        Mockito.verify(sdk).pullModel(any(ModelPullRequest.class), any());
    }
}
