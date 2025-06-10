# vCache Developer Guide

<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/vCache_Logo_For_Dark_Background.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/vCache_Logo_For_Light_Background.png">
    <!-- Fallback -->
    <img alt="vCache" src="./docs/vCache_Logo_For_Dark_Background.png" width="55%">
  </picture>
</p>

<h3 align="center">
Reliable and Efficient Semantic Prompt Caching
</h3>
<br>

This guide provides a quick overview for developers working with vCache. For comprehensive contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## 🚀 Quick Development Setup

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/vcache-project/vCache.git
cd vCache

# Install dependencies
poetry install --with dev,benchmarks

# Set up pre-commit hooks (optional but recommended)
poetry run pre-commit install
```

### Environment Variables

Create a `.env` file for development:

```bash
# Required for integration tests and OpenAI components
OPENAI_API_KEY=your_api_key_here

# Optional development flags
VCACHE_DEBUG=1
VCACHE_LOG_LEVEL=DEBUG
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run only unit tests (fast)
poetry run pytest tests/unit

# Run only integration tests (requires API keys)
poetry run pytest tests/integration

# Run with coverage
poetry run pytest --cov=vcache --cov-report=html
```

## 🔧 Code Quality

```bash
# Format code
poetry run ruff format .

# Lint code
poetry run ruff check . --fix

# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Type checking (if mypy is enabled)
poetry run mypy vcache
```

## 📊 Benchmarking

```bash
# Run benchmarks
poetry run python benchmarks/benchmark.py

# Profile performance
pip install py-spy
py-spy record -o profile.svg -- python your_script.py
```

## 🏗️ Architecture Overview

vCache follows a modular architecture with the following key components:

- **`vcache/main.py`**: Main VCache class and entry point
- **`vcache/config.py`**: Configuration management
- **`vcache/inference_engine/`**: LLM inference backends (OpenAI, Anthropic, etc.)
- **`vcache/vcache_core/`**: Core caching logic
  - **`cache/`**: Embedding storage and vector databases
  - **`similarity_evaluator/`**: Similarity computation strategies
  - **`statistics/`**: Performance tracking
- **`vcache/vcache_policy/`**: Caching decision policies
- **`benchmarks/`**: Performance evaluation framework
- **`tests/`**: Unit and integration tests

## 📝 Quick Coding Guidelines

- **Formatting**: Use `ruff` (88 char line limit)
- **Typing**: Add type annotations to all public APIs
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Imports**: Use absolute imports, organize with ruff
- **Testing**: Add tests for new features and bug fixes
- **Documentation**: Update docstrings and README for API changes

## 🔗 Useful Commands

```bash
# Install new dependency
poetry add package_name

# Install development dependency
poetry add --group dev package_name

# Update dependencies
poetry update

# Build package
poetry build

# Run in poetry environment
poetry run python your_script.py

# Activate poetry shell
poetry shell
```

## 📚 Additional Resources

- [Full Contributing Guide](CONTRIBUTING.md) - Comprehensive development guidelines
- [Test Documentation](tests/ReadMe.md) - Testing framework details
- [Benchmark Documentation](benchmarks/ReadMe.md) - Performance evaluation
- [vCache Paper](https://arxiv.org/abs/2502.03771) - Technical background

For questions or support, please open an issue on [GitHub](https://github.com/vcache-project/vCache/issues).