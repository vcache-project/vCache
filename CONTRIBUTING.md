# Contributing to vCache

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

Thank you for your interest in contributing to vCache! We welcome and value all contributions to the project, including but not limited to:

- [Bug reports](https://github.com/vcache-project/vCache/issues) and [discussions](https://github.com/vcache-project/vCache/discussions)
- [Pull requests](https://github.com/vcache-project/vCache/pulls) for bug fixes and new features
- Test cases to make the codebase more robust
- Examples and benchmarks
- Documentation improvements
- Tutorials, blog posts and talks on vCache

## Contributing code

We use GitHub to track issues and features. For new contributors, we recommend looking at issues labeled ["good first issue"](https://github.com/vcache-project/vCache/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

### Installing vCache for development

Follow the steps below to set up a local development environment for contributing to vCache.

#### Create a Python environment
vCache requires Python 3.10 or higher. We recommend using a virtual environment:

```bash
# Using conda (recommended)
conda create -y -n vcache python=3.11
conda activate vcache

# Or using venv
python3.11 -m venv vcache-env
source vcache-env/bin/activate  # On Windows: vcache-env\Scripts\activate
```

#### Install Poetry
vCache uses [Poetry](https://python-poetry.org/) for dependency management. If you don't have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Install vCache
To install vCache for development, please fork [vcache-project/vCache](https://github.com/vcache-project/vCache) to your GitHub account and run:

```bash
# Clone your forked repo
git clone https://github.com/<your-github-username>/vCache.git

# Set upstream to keep in sync with the official repo
cd vCache
git remote add upstream https://github.com/vcache-project/vCache.git

# Install vCache in editable mode with all dependencies
poetry install --with dev,benchmarks

# Alternatively, install only core dependencies:
# poetry install

# Install development dependencies separately if needed:
# poetry install --with dev
```

#### (Optional) Install `pre-commit`
You can install `pre-commit` hooks to help automatically format your code on commit:

```bash
poetry run pre-commit install
```

### Testing

vCache includes both **unit tests** and **integration tests** to ensure correctness and reliability.

#### Running Unit Tests
Unit tests verify individual module strategies in isolation and are fast and deterministic:

```bash
# Run all unit tests
poetry run pytest tests/unit

# Run specific unit test file
poetry run pytest tests/unit/test_specific_module.py

# Run with verbose output
poetry run pytest tests/unit -v
```

#### Running Integration Tests
Integration tests validate end-to-end behavior and may require API keys:

```bash
# Set up environment variables (create .env file)
echo "OPENAI_API_KEY=your_key_here" > .env

# Run all integration tests
poetry run pytest tests/integration

# Run specific integration test
poetry run pytest tests/integration/test_end_to_end.py

# Terminate test resources on failure (if applicable)
poetry run pytest tests/integration --terminate-on-failure
```

#### Running All Tests
```bash
# Run complete test suite
poetry run pytest

# Re-run last failed tests
poetry run pytest --lf

# Run tests with coverage report
poetry run pytest --cov=vcache --cov-report=html
```

#### Testing in a clean environment
For testing in a clean environment, consider using a fresh virtual environment:

```bash
# Create a clean environment for testing
python -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate
pip install poetry
poetry install --with dev,benchmarks
poetry run pytest
```

### Submitting pull requests

- Fork the vCache repository and create a new branch for your changes.
- If relevant, add tests for your changes. For changes that touch the core system, run the full test suite and ensure tests pass.
- Follow the [coding guidelines](#coding-guidelines) outlined below.
- Ensure code is properly formatted by running `poetry run pre-commit run --all-files`.
- Push your changes to your fork and open a pull request in the vCache repository.
- In the PR description, write a `Tested:` section to describe relevant tests performed.

### Coding Guidelines

To ensure consistency and maintainability across contributions:

#### Code Style and Formatting
- **Formatting**: Use [`ruff`](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`).
- **Line length**: Maximum 88 characters (configured in ruff).
- **Imports**: Use absolute imports. Organize them with ruff's isort compatibility.
- **Quotes**: Use double quotes for strings.

#### Type Annotations
- **Typing**: Follow [PEP 484](https://peps.python.org/pep-0484/) type annotations for all public functions and methods.
- **Type coverage**: Type coverage is checked via `mypy` (when enabled in pre-commit).
- **Import typing**: Import typing-only external objects under `if typing.TYPE_CHECKING:`.

#### Naming Conventions
- **Variables and functions**: Use descriptive, lowercase names with underscores (`snake_case`).
- **Classes**: Use PascalCase for class names.
- **Constants**: Use UPPER_CASE for module-level constants.
- **Private methods**: Prefix with single underscore (`_private_method`).

#### Architecture and Design
- **Modularity**: Keep logic modular and composable—each component (inference engine, policy, vector DB) should follow its defined interface.
- **Single Responsibility**: Each class and function should have a single, well-defined purpose.
- **Interface Compliance**: New strategies should implement the appropriate abstract base class.
- **Dependency Injection**: Use dependency injection patterns for better testability.

### Some general engineering practice suggestions

These are suggestions, not strict rules to follow. When in doubt, follow the established patterns in the codebase.

- Use `TODO(author_name)`/`FIXME(author_name)` instead of blank `TODO/FIXME`. This is critical for tracking down issues.
- Delete your branch after merging it. This keeps the repo clean and faster to sync.
- Use exceptions for error conditions. Only use `assert` for debugging or proof-checking purposes.
- Use lazy imports for heavy third-party modules that are imported during `import vcache` but have significant import time.
- To measure import time:
  - Basic check: `python -X importtime -c "import vcache"`
  - Detailed analysis: use [`tuna`](https://github.com/nschloe/tuna):
    ```bash
    python -X importtime -c "import vcache" 2> import.log
    tuna import.log
    ```
- Use modern Python features that increase code quality:
  - Use f-strings instead of `.format()` for short expressions.
  - Use `class MyClass:` instead of `class MyClass(object):`.
  - Use `abc` module for abstract classes to ensure all abstract methods are implemented.
  - Use context managers (`with` statements) for resource management.

### Component-Specific Guidelines

#### Adding New Inference Engines
When adding a new inference engine:

1. Inherit from `InferenceEngine` abstract base class
2. Implement required methods: `infer()`, `get_model_name()`
3. Add appropriate error handling for API failures
4. Include rate limiting if applicable
5. Add unit tests for the new engine
6. Update documentation with usage examples

#### Adding New Caching Policies
When adding a new caching policy:

1. Inherit from `VCachePolicy` abstract base class
2. Implement required methods: `should_cache()`, `update_statistics()`
3. Ensure thread safety if applicable
4. Add comprehensive unit tests
5. Include performance benchmarks
6. Document the policy's behavior and use cases

#### Adding New Vector Databases
When adding a new vector database:

1. Inherit from appropriate vector DB interface
2. Implement CRUD operations: `add()`, `search()`, `delete()`
3. Handle connection management and error recovery
4. Add integration tests with real data
5. Document setup requirements and configuration options

### Environment variables for developers

- `export VCACHE_DEBUG=1` to enable debug logging.
- `export VCACHE_LOG_LEVEL=DEBUG` to set specific log levels.
- `export VCACHE_DISABLE_TELEMETRY=1` to disable usage analytics (if implemented).
- `export OPENAI_API_KEY=your_key` for OpenAI-based components.
- `export VCACHE_TEST_MODE=1` to enable test-specific behaviors.

### Benchmarking and Performance

When contributing performance improvements:

1. Run existing benchmarks to establish baseline:
   ```bash
   poetry run python benchmarks/benchmark.py
   ```

2. Profile your changes:
   ```bash
   pip install py-spy
   py-spy record -o profile.svg -- python your_test_script.py
   ```

3. Include benchmark results in your PR description.

### Documentation

- Update docstrings for any new or modified public APIs.
- Follow [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Update README.md if adding new features or changing installation procedures.
- Add examples for new functionality.

### Release Process

For maintainers preparing releases:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with new features and fixes
3. Run full test suite: `poetry run pytest`
4. Build and test package: `poetry build`
5. Create release tag: `git tag v0.x.x`
6. Push tag: `git push origin v0.x.x`

Thank you for contributing to vCache! 🚀