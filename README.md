# Tafkir - Java ML/AI Training Framework

[![CI](https://github.com/bhangun/tafkir/actions/workflows/ci.yml/badge.svg)](https://github.com/bhangun/tafkir/actions/workflows/ci.yml)
[![Java](https://img.shields.io/badge/Java-25+-blue.svg)](https://jdk.java.net/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Tafkir** is a high-performance Java-based machine learning training and inference framework, designed to provide PyTorch and scikit-learn-like capabilities for the Java ecosystem. Built on JDK 25+ with support for modern JVM features like Vector API and Panama FFM, Tafkir enables developers to build, train, and deploy ML models entirely in Java.

## 🎯 Key Features

- **Training Framework**: Comprehensive training APIs similar to PyTorch's training loop abstractions
- **Neural Network Modules**: Build custom architectures with layers, activations, and optimizers
- **ONNX Support**: Import and export models via ONNX for interoperability
- **Quantization**: Advanced quantization tools (GPTQ, AWQ, QUIP, TurboQuant) for model optimization
- **Multi-Backend**: Support for CPU, CUDA, Metal, and other accelerators
- **CLI Tools**: Production-ready CLI for model management, inference, and serving
- **JBang Scripts**: Zero-build scripting for rapid prototyping and learning
- **Jupyter Integration**: Notebook support via hybrid Python/Java architecture
- **Vector API**: Leverages JDK incubator Vector API for SIMD acceleration

## 📦 Project Structure

```
tafkir/
├── core/               # Core utilities and base abstractions
├── ml/                 # ML framework APIs (tensors, autograd, nn modules)
├── trainer/            # Training framework and strategies
├── training/           # Training implementations and pipelines
├── tafkir-cli/         # Command-line interface tool (evolved from Gollek)
├── quantizer/          # Model quantization modules
├── compiler/           # Model compilation tools
├── integration/        # Third-party integrations (Wayang, etc.)
├── examples/           # JBang scripts and Jupyter notebooks
└── models/             # Pre-trained model registry
```

**Note on Gollek**: The CLI component evolved from the Gollek inference/serving engine platform. While Gollek continues as a standalone high-performance inference server, Tafkir CLI is now integrated into the Tafkir training framework to provide seamless end-to-end workflows—from model training to deployment.

## 🚀 Quick Start

### Prerequisites

- **JDK 25+** (required for Vector API and Panama FFM features)
- **JBang** (optional, for scripting examples)

### Option 1: JBang (Recommended for Learning)

No build tools needed - just install JBang and run:

```bash
# Install JBang
curl -Ls https://sh.jbang.dev | bash -s -

# Clone and run examples
git clone https://github.com/bhangun/tafkir.git
cd tafkir/examples/jbang
jbang sdk/tafkir-quickstart.java
```

### Option 2: Build from Source

```bash
git clone https://github.com/bhangun/tafkir.git
cd tafkir
./gradlew publishToMavenLocal -x test
```

### Option 3: Use Published Artifacts

Once released, add to your `build.gradle.kts`:

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("tech.kayys.tafkir:tafkir-ml-api:0.3.0")
}
```

## 💻 First Model Example

Create a simple neural network with JBang:

```java
///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-api:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.tafkir.ml.Tafkir;
import tech.kayys.tafkir.ml.tensor.Tensor;
import tech.kayys.tafkir.ml.nn.Linear;
import tech.kayys.tafkir.ml.nn.ReLU;
import tech.kayys.tafkir.ml.nn.Sequential;

public class my_model {
    public static void main(String[] args) {
        // Build a simple neural network
        Sequential model = Sequential.builder()
            .add(new Linear(784, 256))
            .add(new ReLU())
            .add(new Linear(256, 10))
            .build();

        // Create random input (simulating a 28x28 image)
        Tensor input = Tafkir.randn(1, 784);

        // Forward pass
        Tensor output = model.forward(input);

        System.out.println("Model: 784 → 256 → 10");
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
    }
}
```

Run it:
```bash
jbang my_model.java
```

## 📚 Documentation

- **[Quickstart Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Architecture Overview](ARCHITECTURE.md)** - Tafkir & Gollek relationship explained
- **[JBang Examples](examples/jbang/INDEX.md)** - Complete catalog of runnable examples
- **[Jupyter Notebooks](examples/jupyter/README.md)** - Interactive ML workflows
- **[CLI Documentation](tafkir-cli/README.md)** - Command-line tool reference
- **[Setup Guide](examples/docs/SETUP.md)** - Detailed installation instructions

## 🔧 CLI Usage

The Tafkir CLI provides production-ready model management and inference:

```bash
# Run inference
tafkir run --model qwen2.5-7b-instruct-GGUF --prompt "Hello"

# Interactive chat
tafkir chat --model llama-3.2-1b-instruct

# List local models
tafkir list

# Download model
tafkir pull hf:Qwen/Qwen2.5-0.5B-Instruct

# Convert to GGUF
tafkir convert --input ~/models/llama-2-7b --output ~/conversions --quant q4_k_m
```

## 🎓 Example Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **SDK** | Core SDK usage patterns | quickstart, tensor ops, export |
| **Trainer** | Training framework demos | runtime bootstrap, quality gates |
| **Neural Networks** | Custom model architectures | MLP, CNN, RNN examples |
| **NLP** | Natural language processing | sentiment analysis, transformers |
| **Vision** | Computer vision models | image classification, detection |
| **Multimodal** | Multi-modal AI | text-to-image, VLM |
| **Quantization** | Model compression | GPTQ, AWQ, QUIP examples |
| **Integration** | Third-party bridges | Wayang, agent systems |

## 🏗️ Architecture Highlights

### Training Framework
- Profile-driven training with budget constraints
- Quality-aware CI/CD gates
- Diffusion model support (Stable Diffusion, DDIM)
- Listener-based lifecycle management

### Inference Engine
- Format-aware provider routing (GGUF, SafeTensors, TorchScript)
- Unified model registry
- GPU acceleration (CUDA, Metal)
- Streaming and batch inference modes

### Quantization Pipeline
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Multiple algorithms: GPTQ, AWQ, QUIP, TurboQuant

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up your development environment
- Code style and conventions
- Submitting pull requests
- Reporting bugs and requesting features

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🗺️ Roadmap

- [ ] Publish stable releases to Maven Central
- [ ] Expand model zoo with pre-trained models
- [ ] Add distributed training support
- [ ] Enhance Jupyter kernel integration
- [ ] More comprehensive documentation and tutorials

## 🙏 Acknowledgments

Tafkir builds upon excellent open-source projects including:
- llama.cpp for GGUF inference
- ONNX Runtime for model interoperability
- Quarkus for native compilation
- JBang for Java scripting

---

**Built with ❤️ for the Java AI community**

For questions or discussions, please open an issue on GitHub.
