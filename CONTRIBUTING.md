# Contributing to Tafkir

Thank you for your interest in contributing to Tafkir! We welcome contributions from the community to help build a robust Java-based ML/AI training framework.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

Please be respectful and constructive in your interactions. We're building an inclusive community for Java developers interested in machine learning.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/tafkir.git
   cd tafkir
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/bhangun/tafkir.git
   ```

## Development Setup

### Prerequisites

- **JDK 25+** (required - Tafkir uses Vector API and Panama FFM)
- **Gradle 8.x** (or use the included Gradle wrapper)
- **Git**

### Build the Project

```bash
# Build and publish to local Maven repository
./gradlew publishToMavenLocal -x test

# Run tests
./gradlew test

# Build specific module
./gradlew :ml:tafkir-ml-api:build
```

### IDE Setup

Tafkir works with any Java IDE:
- **IntelliJ IDEA** (recommended) - Enable preview features for JDK 25
- **Eclipse** - Configure JDK 25 with preview features
- **VS Code** - With Extension Pack for Java

Enable these JVM flags for development:
```
--enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## How to Contribute

### Ways to Contribute

1. **Bug Reports**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Propose new features via GitHub issues
3. **Documentation**: Improve docs, add examples, fix typos
4. **Code Contributions**: Fix bugs or implement features
5. **Testing**: Add test cases or improve test coverage

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Need community assistance
- `bug` - Bugs to fix
- `enhancement` - Features to implement

## Coding Standards

### General Guidelines

- Follow standard Java naming conventions
- Keep methods small and focused (single responsibility)
- Write meaningful comments for complex logic
- Use meaningful variable and method names
- Prefer immutability where possible

### Code Style

```java
// Good: Clear method name, focused responsibility
public Tensor forward(Tensor input) {
    return layers.stream()
        .reduce(input, (acc, layer) -> layer.apply(acc), (a, b) -> b);
}

// Avoid: Vague names, multiple responsibilities
public Tensor f(Tensor t) {
    // ... 100 lines of mixed logic
}
```

### Documentation

- Add Javadoc for public APIs
- Include usage examples in Javadoc
- Update README.md for user-facing changes
- Document breaking changes clearly

Example:
```java
/**
 * Applies a linear transformation to the input tensor.
 * 
 * <p>Computes: output = input * weight^T + bias</p>
 *
 * @param input the input tensor of shape (batch, inFeatures)
 * @return the output tensor of shape (batch, outFeatures)
 * @throws IllegalArgumentException if input dimensions don't match
 */
public Tensor forward(Tensor input);
```

## Testing

### Running Tests

```bash
# Run all tests
./gradlew test

# Run specific module tests
./gradlew :ml:tafkir-ml-api:test

# Run with coverage
./gradlew jacocoTestReport
```

### Writing Tests

- Use JUnit 5 for unit tests
- Use AssertJ for fluent assertions
- Test edge cases and error conditions
- Aim for high test coverage on critical paths

Example:
```java
@Test
void testLinearForwardPass() {
    Linear layer = new Linear(784, 256);
    Tensor input = Tafkir.randn(32, 784);
    
    Tensor output = layer.forward(input);
    
    assertThat(output.shape())
        .containsExactly(32, 256);
    assertThat(output.dtype())
        .isEqualTo(DType.FLOAT32);
}
```

### CI/CD

All pull requests must pass:
- Unit tests
- Integration tests
- Code style checks
- Build verification

The CI pipeline runs automatically on PRs via GitHub Actions.

## Submitting Changes

### Branch Naming

Use descriptive branch names:
```
feature/add-transformer-layer
fix/tensor-shape-validation
docs/update-quickstart
test/add-autograd-tests
```

### Commit Messages

Follow conventional commit format:
```
feat: add RMSNorm layer implementation

Implement RMSNorm (Root Mean Square Layer Normalization) 
as an alternative to LayerNorm for transformer models.

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a PR** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues
   - Screenshots/logs if applicable

3. **Address feedback** from reviewers

4. **Squash commits** if requested before merge

### PR Checklist

Before submitting your PR, ensure:
- [ ] Code follows project style guidelines
- [ ] Tests are added/updated
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear and conventional
- [ ] Branch is up-to-date with main

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Check repository maintainers for contact info

### Recognition

Contributors will be:
- Listed in the CONTRIBUTORS file
- Mentioned in release notes for significant contributions
- Recognized in project documentation

---

**Thank you for contributing to Tafkir!** 🎉

Your contributions help make Java a first-class citizen in the ML/AI ecosystem.
