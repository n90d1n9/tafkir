# Tafkir & Gollek Architecture Overview

## Project Evolution

This document explains the relationship between **Tafkir** and **Gollek**, two related but distinct platforms in our ML/AI ecosystem.

### Historical Context

The project originated as **Gollek**, a high-performance inference and serving engine for large language models. As the platform matured, the training capabilities grew into a comprehensive machine learning framework, which was spun off as **Tafkir**.

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML/AI Ecosystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │     Gollek       │         │     Tafkir       │        │
│  │                  │         │                  │        │
│  │  Inference &     │         │  Training        │        │
│  │  Serving Engine  │         │  Framework       │        │
│  │                  │         │                  │        │
│  │  - GGUF Runtime  │         │  - Neural Nets   │        │
│  │  - SafeTensors   │         │  - Autograd      │        │
│  │  - Model Serving │         │  - Training Loop │        │
│  │  - CLI Tools     │◄───────►│  - Optimizers    │        │
│  │  - Quantization  │         │  - Data Loading  │        │
│  └──────────────────┘         │  - Strategies    │        │
│                               └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Platform Comparison

| Feature | Gollek | Tafkir |
|---------|--------|--------|
| **Primary Focus** | Inference & Serving | Training & Development |
| **Use Case** | Deploy models to production | Build and train custom models |
| **Package Namespace** | `tech.kayys.gollek.*` | `tech.kayys.tafkir.*` |
| **CLI** | ✅ Gollek CLI | ✅ Tafkir CLI (shared heritage) |
| **Model Support** | Pre-trained models | Custom + pre-trained |
| **Quantization** | ✅ PTQ (GPTQ, AWQ, etc.) | ✅ QAT + PTQ |
| **Training APIs** | ❌ | ✅ PyTorch-like |
| **Jupyter Support** | Limited | ✅ Full integration |
| **JBang Scripts** | Inference examples | Training + inference |

### Shared Components

Both platforms share several components due to their common heritage:

1. **CLI Tool**: The command-line interface evolved from Gollek and is now integrated into Tafkir for end-to-end workflows
2. **Model Registry**: Shared model discovery and management
3. **Quantization Modules**: GPTQ, AWQ, QUIP, TurboQuant implementations
4. **Format Support**: GGUF, SafeTensors, TorchScript, ONNX

### Package Structure

#### Gollek Packages (`tech.kayys.gollek.*`)
- Model family plugins (e.g., `tafkir-model-llama`, `tafkir-model-bert`)
- Inference runtime providers
- Serving infrastructure
- Legacy SPI implementations

#### Tafkir Packages (`tech.kayys.tafkir.*`)
- `tafkir.ml.*` - Core ML abstractions (tensors, autograd, nn modules)
- `tafkir.trainer.*` - Training framework APIs
- `tafkir.training.*` - Training implementations
- `tafkir.sdk.*` - High-level SDK for developers
- `tafkir.cli.*` - Command-line interface

### When to Use Which

**Use Gollek when:**
- You need to deploy pre-trained models to production
- You want maximum inference performance
- You're building serving infrastructure
- You need quantization for existing models

**Use Tafkir when:**
- You want to train custom models from scratch
- You need PyTorch-like training workflows in Java
- You're experimenting with new architectures
- You want Jupyter notebook support
- You need end-to-end ML pipelines (train → evaluate → deploy)

### Integration Example

A typical workflow might use both platforms:

```java
// Step 1: Train custom model with Tafkir
Trainer trainer = Trainers.canonicalBuilder()
    .model(myCustomArchitecture)
    .dataset(trainingData)
    .build();
trainer.fit();

// Step 2: Export to ONNX or SafeTensors
ModelExporter.export(trainer.getModel(), "model.safetensors");

// Step 3: Deploy with Gollek for high-performance serving
// (or use Tafkir CLI which includes Gollek inference capabilities)
```

### Migration Path

For existing Gollek users:
- Model plugins under `tech.kayys.gollek.*` continue to work
- New training features should use `tech.kayys.tafkir.*` packages
- CLI commands are compatible across both platforms
- Gradle module names use `tafkir-*` prefix for consistency

### Version Compatibility

| Component | Version Scheme | Release Status |
|-----------|---------------|----------------|
| Tafkir Core | 0.3.0-SNAPSHOT | Active development |
| Gollek Inference | 1.x.x | Stable |
| Shared CLI | Unified | Latest in Tafkir |
| Model Plugins | Varies by model | Mixed |

### Future Direction

- **Tafkir** will continue expanding training capabilities, adding more PyTorch/scikit-learn parity
- **Gollek** will focus on inference optimization and serving performance
- Both platforms will maintain compatibility through shared formats (ONNX, SafeTensors, GGUF)
- The CLI will remain unified for seamless train-to-deploy workflows

---

**Questions?** Open an issue on GitHub or check the main [README.md](../README.md) for getting started guides.
