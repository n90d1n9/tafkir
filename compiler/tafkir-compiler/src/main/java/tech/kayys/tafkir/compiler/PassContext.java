package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.OpSchemaRegistry;

public final class PassContext {
    public final OpSchemaRegistry schemaRegistry;

    public PassContext(OpSchemaRegistry schemaRegistry) {
        this.schemaRegistry = schemaRegistry;
    }
}