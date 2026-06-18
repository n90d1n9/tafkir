package tech.kayys.tafkir.compiler.rewrite;

import tech.kayys.tafkir.ir.GGraph;

public final class RewriteResult {
    public final GGraph graph;
    public final boolean changed;

    public RewriteResult(GGraph graph, boolean changed) {
        this.graph = graph;
        this.changed = changed;
    }
}
