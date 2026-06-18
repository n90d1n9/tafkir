package tech.kayys.tafkir.training.strategy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public class FullParameterStrategy implements FineTuningStrategy {

    @Override
    public MemorySegment computeGradientsAndUpdate(MemorySegment contextSegment, MemorySegment modelSegment, Arena arena) {
        // Placeholder for standard backpropagation across all weights
        System.out.println("Executing Full Parameter training step");
        return MemorySegment.NULL;
    }
}
