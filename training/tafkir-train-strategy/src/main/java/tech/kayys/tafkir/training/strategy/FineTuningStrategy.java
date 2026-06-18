package tech.kayys.tafkir.training.strategy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Abstraction for various fine-tuning methodologies natively operating on zero-copy memory.
 */
public interface FineTuningStrategy {

    /**
     * Executes a training step directly on off-heap memory segments.
     *
     * @param contextSegment The training data/context (zero-copy from Aljabr)
     * @param modelSegment   The current model or adapter weights (zero-copy from Aljabr)
     * @param arena          The memory lifecycle arena backing this request
     * @return A MemorySegment containing the updated weights or adapter
     */
    MemorySegment computeGradientsAndUpdate(MemorySegment contextSegment, MemorySegment modelSegment, Arena arena);
}
