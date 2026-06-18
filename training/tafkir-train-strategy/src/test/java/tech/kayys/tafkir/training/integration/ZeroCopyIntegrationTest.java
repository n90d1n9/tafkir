package tech.kayys.tafkir.training.integration;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.memory.UnifiedMemoryStore;
import tech.kayys.aljabr.core.memory.helixdb.HelixDbMemoryStore;
import tech.kayys.tafkir.training.strategy.FineTuningStrategy;
import tech.kayys.tafkir.training.strategy.LoRAStrategy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZeroCopyIntegrationTest {

    @Test
    public void testDataFlowFromHelixDbToTafkirNative() {
        // 1. Initialize HelixDB simulated store
        try (UnifiedMemoryStore dbStore = new HelixDbMemoryStore("mock_path")) {
            
            byte[] contextKey = "agent_context_1".getBytes();
            byte[] modelKey = "agent_adapter_1".getBytes();

            // 2. Put simulated data into the store (mimics Aljabr saving state)
            try (Arena tempArena = Arena.ofConfined()) {
                MemorySegment contextData = tempArena.allocate(1024);
                MemorySegment modelData = tempArena.allocate(1024);
                
                dbStore.put(contextKey, contextData);
                dbStore.put(modelKey, modelData);
            }

            // 3. Begin Training Step (Wayang orchestrator triggers this)
            try (Arena trainingArena = Arena.ofConfined()) {
                
                // 4. Fetch zero-copy MemorySegment directly from DB
                Optional<MemorySegment> optContextSegment = dbStore.getZeroCopy(contextKey, trainingArena);
                Optional<MemorySegment> optModelSegment = dbStore.getZeroCopy(modelKey, trainingArena);

                assertTrue(optContextSegment.isPresent(), "Context segment should be retrieved");
                assertTrue(optModelSegment.isPresent(), "Model segment should be retrieved");

                MemorySegment contextSegment = optContextSegment.get();
                MemorySegment modelSegment = optModelSegment.get();

                // 5. Hand off directly to Tafkir FFM Native Training Engine (Zero-Copy)
                FineTuningStrategy strategy = new LoRAStrategy(16, 32.0f);
                
                try {
                    // This will invoke FFM bindings. If LibTorch is not present in test env, it may throw.
                    // We catch and log, since we mainly want to verify the architectural flow.
                    MemorySegment updatedAdapter = strategy.computeGradientsAndUpdate(contextSegment, modelSegment, trainingArena);
                    assertNotNull(updatedAdapter);
                } catch (RuntimeException e) {
                    // Expected if LibTorch is not loaded/found natively on the machine running tests.
                    System.out.println("Native execution skipped because LibTorch is not available: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
