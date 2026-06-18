package tech.kayys.tafkir.compiler.rewrite;

import tech.kayys.tafkir.ir.*;

public interface RewriteRule {
    boolean matches(GGraph graph, int index);

    RewriteResult apply(GGraph graph, int index);
}