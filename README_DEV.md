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

## Coding Guidelines
