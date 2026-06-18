package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.GGraph;

/**
 * COMPILER GOAL
 * Transform:
 * High-level IR (attention, ffn, etc.)
 * Optimized IR (fused ops, layout-aware)
 * ↓
 * ↓
 * Executable graph
 */
public interface Pass {
    String name();

    GGraph apply(GGraph graph, PassContext ctx);
}