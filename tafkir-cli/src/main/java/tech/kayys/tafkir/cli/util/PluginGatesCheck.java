/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.tafkir.cli.util;

import java.util.Map;

/**
 * Build-time smoke check for all Tafkir plugin release gates.
 */
public final class PluginGatesCheck {
    private PluginGatesCheck() {
    }

    public static void main(String[] args) throws Exception {
        try (ExternalPluginClasspathScope pluginScope =
                ExternalPluginClasspathScope.open(args, 0, PluginGatesCheck.class)) {
            Map<String, Object> report = PluginGatesReportWriter.buildReport(
                    pluginScope.discoveryClassLoader(),
                    pluginScope.classpath());
            PluginGates gates = PluginGatesReportWriter.gatesFromReport(report);
            if (gates.failed()) {
                throw new IllegalStateException(gates.failureMessage());
            }
            System.out.printf(
                    "Tafkir plugin gates %s (%s violation(s))%n",
                    gates.status(),
                    gates.violationCount());
        }
    }
}
