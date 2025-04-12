# Tests

## Unit Tests
The unit tests are supposed to soley test the logic of an invidual module strategy.

## Integration Tests
The integration tests are supposed to test the combination and interaction of all module strategies.

## Run All Tests

```bash
pip install -e .
```

```bash
export OPENAI_API_KEY="your_api_key_here"
```

```bash
python3 runner.py
```

## Run Individual Tests

```bash
pytest unit/VectorDBStrategy/test.py
```

With print terminal output enabled
```bash
pytest -vs unit/VectorDBStrategy/test.py
```