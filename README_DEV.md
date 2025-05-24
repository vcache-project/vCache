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


This guide outlines how to set up a development environment for vCache, enforce code quality, and follow coding standards.

## 🧑‍💻 Developer Guide Beginning

### Environment Setup with Poetry

vCache uses [Poetry](https://python-poetry.org/) for dependency management.

#### 1. Install Poetry

If you don’t have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### 2. Install Dependencies

Install all necessary packages for development and benchmarking:

```bash
poetry install --with dev,benchmarks
```

This will set up the base environment along with development tools and benchmark dependencies.



### Setting Up Pre-commit Hooks

We use `pre-commit` to maintain code quality.

#### Install Hooks

```bash
poetry run pre-commit install
```

#### What It Checks

On every commit, the following checks will be run automatically:
- Code formatting with **Ruff**
- Import order validation
- Python syntax validation
- Static type checking with **mypy**

#### Run Checks Manually

You can manually trigger all pre-commit hooks with:

```bash
poetry run pre-commit run --all-files
```

## 🧪 Running Tests

See the [Test Documentation](tests/ReadMe.md) for instructions.


## 📏 Coding Guidelines

To ensure consistency across contributions:

- **Formatting**: Use [`ruff`](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`).
- **Typing**: Follow [PEP 484](https://peps.python.org/pep-0484/) type annotations. Type coverage is checked via `mypy`.
- **Structure**: Keep logic modular and composable—each component (inference, policy, vector DB) should follow its defined interface.
- **Imports**: Absolute imports are preferred. Organize them with `ruff`’s isort compatibility.
- **Naming**: Use descriptive, lowercase names with underscores for variables and functions; PascalCase for class names.

