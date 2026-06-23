# Tafkir Integration Roadmap

## Executive Summary

**Tafkir is a thin compatibility layer over Aljabr.** The current `ml/tafkir-ml-autograd` module contains duplicated, inferior implementations that should be replaced with direct Aljabr backend usage.

This document outlines the phased integration plan to make Tafkir a working ML framework by leveraging Aljabr's compute engine.

---

## Current State Analysis

### What Aljabr Has (n90d1n9/aljabr)

| Module | Capabilities |
|--------|-------------|
| `core/aljabr-tensor` | `Tensor`, `Shape`, `DType`, `DeviceType` abstractions |
| `core/aljabr-core` | `ManagedArena` (ref-counted Panama FFM memory), `UnifiedMemoryStore` |
| `core/aljabr-nn` | `Linear`, `Sequential`, activation functions |
| `autograd/` | `AutogradEngine`, `GradRegistry`, `GGraph` topological sort, 12+ gradient functions |
| `backend/cpu/` | `CpuBackend` with Vector API SIMD, matmul, conv2d, attention, normalization |
| `backend/cuda/` | `CudaBackend` with CUDA kernel bindings |
| `backend/metal/` | Metal backend for Apple Silicon |
| `backend/blackwell/` | `BlackwellRunner` with FlashAttention-3, FP4, TMEM for transformer inference |

### What Tafkir Re-Invented (Poorly)

| Tafkir Component | Problem | Quality Gap |
|-----------------|---------|-------------|
| `GradTensor.java` | Array-backed `float[]` on Java heap | ~50-100x slower than Vector API, GC pressure |
| `TensorOps.einsum()` | 2 hardcoded equations with 7-deep nested loops | No general einsum, not extensible |
| `Function.java` | Single `backward()` callback, no graph | No topological sort, broken for branched graphs |
| `CanonicalTrainerRuntime` | Fake training loop with synthetic loss | Doesn't actually train models |
| No `Device` abstraction | Claims "multi-backend" but CPU-only | No GPU training possible |

---

## Phase 1: Stop Re-Inventing — Use Aljabr as the Engine (2-3 weeks)

### Step 1.1: Create TafkirTensor Wrapper ✓

**Status:** Complete

Created `/workspace/ml/tafkir-ml-autograd/src/main/java/tech/kayys/tafkir/ml/tensor/TafkirTensor.java`

This is a thin wrapper that will delegate to Aljabr's `Tensor` and `ComputeBackend`. Currently uses reflection placeholders until Aljabr is properly linked.

**Next action:** Replace reflection with direct method calls once Aljabr dependency is resolved.

### Step 1.2: Delete Legacy Tensor Code

**Files to delete:**
- `ml/tafkir-ml-autograd/src/main/java/tech/kayys/tafkir/ml/autograd/GradTensor.java`
- `ml/tafkir-ml-autograd/src/main/java/tech/kayys/tafkir/ml/autograd/TensorOps.java`
- `ml/tafkir-ml-autograd/src/main/java/tech/kayys/tafkir/ml/autograd/Function.java`

**Replacement:** All code should use `TafkirTensor` which delegates to Aljabr.

### Step 1.3: Wire Aljabr's AutogradEngine

Replace the current backward pass mechanism with Aljabr's graph-based autograd:

```java
// New pattern
GGraph forwardGraph = model.forwardGraph(input);
Tensor pred = forwardGraph.outputs().get(0);
Tensor lossValue = loss.compute(pred, target);
GGraph backwardGraph = autograd.buildBackward(forwardGraph, lossValue.id());
backend.execute(backwardGraph);
```

---

## Phase 2: Fix the Trainer to Actually Train (2-3 weeks)

### Current Problems

`CanonicalTrainerRuntime` has:
- `model(Object ignored)` — no-op, doesn't accept a real model
- `optimizer(Object ignored)` — no-op
- `loss(Object candidate)` — accepts lambdas, not real loss functions
- `syntheticLoss(int seed)` — fake loss when no evaluator provided

### Target API

```java
public interface TafkirModel {
    GGraph forward(GGraph input);
    List<Tensor> parameters();
}

public interface TafkirLoss {
    Tensor compute(Tensor pred, Tensor target);
}

public interface TafkirOptimizer {
    void step(List<Tensor> parameters);
}
```

### Implementation Tasks

1. **Create `TafkirSequential`** — wraps Aljabr's `Sequential` or builds `GGraph` from layers
2. **Create `TafkirCrossEntropyLoss`** — uses `backend.crossEntropy()`
3. **Create `TafkirAdam`** — implements Adam using backend tensor ops
4. **Rewrite `CanonicalTrainerRuntime.fit()`** — actual forward → loss → backward → step loop

---

## Phase 3: Unify Namespace and Build System (1 week)

### Package Name Fixes

| Old | New |
|-----|-----|
| `tech.kayys.gollek.models.*` | `tech.kayys.aljabr.models.*` or `tech.kayys.tafkir.models.*` |
| `tech.kayys.tafkir.ml.autograd.GradTensor` | DELETE → use `tech.kayys.aljabr.tensor.Tensor` |

### Build Configuration Fixes

**`build.gradle.kts`:**
```kotlin
// Change this:
url.set("https://github.com/bhangun/tafkir ")
// To this:
url.set("https://github.com/n90d1n9/tafkir ")
```

**`settings.gradle.kts`:**
```kotlin
// Already correct, but add documentation
includeBuild("../aljabr") // Required: Aljabr is the compute engine
```

### CI Configuration

Add to `.github/workflows/ci.yml`:
```yaml
- name: Checkout Aljabr dependency
  run: |
    git clone https://github.com/n90d1n9/aljabr.git ../aljabr
```

---

## Phase 4: Make CLI Use Aljabr for Inference (1-2 weeks)

### Current State

`tafkir-cli` uses `llama.cpp` via Panama FFM for GGUF inference only. Claims to run SafeTensors and TorchScript but can't.

### Target State

```java
// In FormatAwareProviderRouter
if (format == ModelFormat.SAFETENSORS) {
    // Load weights into Aljabr Tensor
    // Run via CpuBackend or CudaBackend
}

@Inject
BlackwellRunner blackwellRunner; // From Aljabr

public InferenceResponse runSafeTensors(ModelManifest manifest, InferenceRequest req) {
    blackwellRunner.initialize(manifest, config);
    return blackwellRunner.infer(req);
}
```

---

## Phase 5: Fix Examples (2-3 days)

### hello_tafkir.java

**Current (broken):**
```java
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
Sequential model = new Sequential(...); // Doesn't exist
```

**Fixed:**
```java
//DEPS tech.kayys.aljabr:aljabr-nn:0.1.0
//DEPS tech.kayys.aljabr:aljabr-backend-cpu:0.1.0
import tech.kayys.aljabr.nn.*;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

CpuBackend backend = new CpuBackend();
Sequential model = new Sequential(
    new Linear(10, 5),
    new ReLU(),
    new Linear(5, 1)
);
Tensor input = backend.randn(1, 10);
Tensor output = model.forward(input);
```

### mnist_style_setup.java

Rewrite to use Aljabr's `CpuBackend`, `AutogradEngine`, and real trainer.

---

## Phase 6: Testing Strategy (2-3 weeks)

### Required Tests

| Test | Verifies |
|------|----------|
| `TensorCreationTest` | `zeros()`, `ones()`, `randn()` create correct shapes |
| `TensorOpTest` | `matmul`, `add`, `mul` match NumPy reference |
| `AutogradTest` | `x + x` backward gives `2`, gradients correct |
| `XORTrainingTest` | 2-layer MLP achieves < 0.01 loss on XOR |
| `MNISTTrainingTest` | Small CNN achieves > 90% accuracy |
| `CheckpointTest` | Resume from checkpoint continues training |
| `BackendParityTest` | CPU and CUDA produce same outputs (within epsilon) |

---

## Phase 7: Documentation Overhaul (1 week)

### Honest README

```markdown
# Tafkir — Java ML/AI Framework

Tafkir is a high-level API for machine learning in Java, built on top of the 
Aljabr compute engine. It provides PyTorch-like training APIs and a CLI for 
model inference.

## Architecture

- **Tafkir (this repo)** = High-level APIs, CLI, examples, quantizer configs
- **Aljabr (../aljabr)** = Compute engine: tensor ops, autograd, CPU/CUDA/Metal backends

## Current Status

- [x] Tensor operations via Aljabr CPU backend (Vector API SIMD)
- [x] Autograd via Aljabr graph engine
- [x] LLM inference via Aljabr Blackwell runner / llama.cpp
- [x] GGUF/SafeTensors model loading
- [ ] End-to-end training loop (in progress)
- [ ] CUDA training backend
- [ ] Distributed training

## Quick Start

# Requires JDK 25+ and Aljabr checked out at ../aljabr
git clone https://github.com/n90d1n9/aljabr.git ../aljabr
./gradlew :tafkir-cli:quarkusBuild
```

---

## Summary: The Single Most Important Fix

**Tafkir's `ml/tafkir-ml-autograd` module is dead weight.** It duplicates Aljabr's functionality with a broken implementation.

**The fix:**

1. ✅ **Created** `TafkirTensor.java` — thin wrapper over Aljabr
2. ⏳ **Delete** `GradTensor.java`, `TensorOps.java`, `Function.java`
3. ⏳ **Replace** `CanonicalTrainerRuntime`'s fake loop with real Aljabr-backed training
4. ⏳ **Update** all examples to use `tech.kayys.aljabr.*` classes

This is a **rewrite, not a refactor**. But it's the only path to a working framework.

---

## Immediate Next Steps

1. **Clone Aljabr** at `../aljabr` to enable compilation
2. **Update `build.gradle.kts`** dependencies to reference actual Aljabr artifacts
3. **Replace reflection in `TafkirTensor`** with direct method calls
4. **Write first integration test** verifying `TafkirTensor.matmul()` produces correct results
5. **Begin `CanonicalTrainerRuntime` rewrite** with real forward/backward passes
