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


**vCache** is the first verified semantic cache with user-defined failure rate guarantees. It employs an online learning algorithm to estimate an optimal threshold for each cached prompt, enabling reliable cache responses without additional training.

## Quick Install

First, install the vCache package.
```bash
pip install -e .
```

Second, set the OpenAI key. By default, vCache uses OpenAI for LLM inference and embedding generation, but you can configure any inference setting you like. 
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Third, use vCache for your LLM inference.
```python
from vcache.main import vCache

vcache: vCache = vCache()
response, cache_hit = vcache.create("Is the sky blue?")

print(f"Response: {response}")
```

## Development Setup

To set up vCache for development:

### Using Poetry

1. Install Poetry if you don't have it already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install --with dev,benchmarks
```

### Setting Up Pre-commit Hooks

Install pre-commit hooks to ensure code quality:
```bash
poetry run pre-commit install
```

The pre-commit hooks will automatically:
- Format code with Ruff
- Check imports
- Validate Python syntax
- Run type checking with mypy

When you commit changes, these checks will run automatically. You can also run them manually:
```bash
poetry run pre-commit run --all-files
```

## Semantic Prompt Caches
Semantic caches return cached LLM-generated responses for semantically similar prompts to reduce inference latency and cost. They embed cached prompts and store them alongside their response in a vector database. Embedding similarity metrics assign a numerical score to quantify the similarity between a request and its nearest neighbor prompt from the cache.

## Benchmarking vCache

vCache includes a benchmarking framework to evaluate performance metrics such as cache hit rates, error rates, and latency improvements. For detailed instructions on running benchmarks, see the [Benchmarking Documentation](benchmarks/README.md).
