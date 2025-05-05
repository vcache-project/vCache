# Tests

## Unit Tests
The unit tests are supposed to soley test the logic of an invidual module strategy.

## Integration Tests
The integration tests are supposed to test the combination and interaction of all module strategies.

### Run Integration Tests
Set `OPEN_AI_APIKEY` in `.env`, and run:

```base
poetry run pytest tests/integration
```