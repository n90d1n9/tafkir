# Tafkir

A Java machine learning framework built on the [Aljabr](https://github.com/bhangun/aljabr) compute engine.

## Architecture

```
┌─────────────────────────────────────────┐
│              Tafkir                     │
│  (training APIs, CLI, examples)       │
├─────────────────────────────────────────┤
│           Aljabr Engine                 │
│  (tensor ops, autograd, backends)     │
│  • CPU: Vector API SIMD                 │
│  • CUDA: GPU kernels                    │
│  • Metal: Apple Silicon                 │
│  • Blackwell: NVIDIA B100/B200          │
└─────────────────────────────────────────┘
```

## Prerequisites

- JDK 25+
- Aljabr checked out at `../aljabr` (composite build)

## Quick Start

```bash
git clone https://github.com/bhangun/aljabr.git ../aljabr
git clone https://github.com/bhangun/tafkir.git
cd tafkir
./gradlew :trainer:tafkir-trainer-aljabr:test
```

## Status

| Feature | Status |
|---------|--------|
| Tensor ops (CPU, SIMD) | Working |
| Autograd (graph-based) | Working |
| MLP training | Working |
| CNN (conv2d, pool) | Working |
| Data loaders (MNIST) | Working |
| LLM inference (GGUF) | Working (via llama.cpp) |
| CUDA training | In progress |
| Distributed training | Planned |

## Example: XOR

```java
TafkirSequential model = new TafkirSequential(
    new TafkirLinear(2, 2),
    new TafkirReLU(),
    new TafkirLinear(2, 1)
);

TafkirTrainer trainer = new TafkirTrainer(
    model,
    new TafkirMSELoss(),
    new TafkirAdam(model.parameters(), 0.1f),
    1000
);

trainer.fit(x, y);
```

## Example: MNIST CNN

```java
TafkirSequential model = new TafkirSequential(
    new TafkirConv2d(1, 32, 3, 1, 1, 1, 1),
    new TafkirReLU(),
    new TafkirMaxPool2d(2, 2, 0),
    new TafkirConv2d(32, 64, 3, 1, 1, 1, 1),
    new TafkirReLU(),
    new TafkirMaxPool2d(2, 2, 0),
    new TafkirFlatten(),
    new TafkirLinear(3136, 128),
    new TafkirReLU(),
    new TafkirLinear(128, 10)
);
```

## Modules

| Module | Description |
|--------|-------------|
| `ml:tafkir-ml-aljabr` | Tensor API wrapping Aljabr backends |
| `trainer:tafkir-trainer-aljabr` | Training loop, optimizers, losses |
| `data:tafkir-data` | Dataset loaders (MNIST, etc.) |
| `distributed:tafkir-distributed` | Multi-threaded data parallel training |
| `checkpoint:tafkir-checkpoint` | Model save/load |
| `tafkir-cli` | Command-line inference tool |
