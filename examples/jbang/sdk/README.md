# Tafkir JBang SDK Examples

This folder contains ML framework and SDK-facing JBang examples.

During migration, this folder intentionally includes two lanes:

- compatibility examples that still use older `tafkir-sdk-*` naming
- newer examples that align better with the canonical module split

## Recommended Order

1. `tafkir-quickstart.java`
2. `tafkir-sdk-core-example.java`
3. `tafkir-sdk-train-example.java`
4. `tafkir-sdk-vision-example.java`
5. `tafkir-sdk-export-example.java`
6. `tafkir-sdk-augment-example.java`

Run from `tafkir/examples/jbang`:

```bash
jbang sdk/tafkir-quickstart.java
jbang sdk/tafkir-sdk-core-example.java
jbang sdk/tafkir-sdk-train-example.java 2
jbang sdk/tafkir-sdk-vision-example.java
jbang sdk/tafkir-sdk-export-example.java
jbang sdk/tafkir-sdk-augment-example.java
```

## Additional Samples

Compatibility-named files now rewritten as runnable scripts:

- `tafkir-quickstart.java`
- `tafkir-sdk-core-example.java`
- `tafkir-sdk-vision-example.java`
- `tafkir-sdk-train-example.java`
- `tafkir-sdk-augment-example.java`
- `tafkir-sdk-export-example.java`

Exploratory v0.3-style demos:

- `unified_framework_demo.java`
- `graph_fusion_example.java`

Legacy v0.2/v0.3 references:

- `tensor_operations_v02.java`
- `vision_transforms_v02.java`
- `tokenization_v02.java`
- `mnist_training_v02.java`
- `pytorch_comparison_v02.java`

Treat these as evolving references rather than the canonical trainer/runtime
surface.

## Prerequisites

From `tafkir/` project root:

```bash
./run-install-local-macos.sh
```

Or publish only the modules a specific script needs with targeted
`publishToMavenLocal` tasks.

Then run with JBang:

```bash
jbang --fresh sdk/tafkir-quickstart.java
```
