<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./../docs/vCache_Logo_For_Dark_Background.png">
    <source media="(prefers-color-scheme: light)" srcset="./../docs/vCache_Logo_For_Light_Background.png">
    <!-- Fallback -->
    <img alt="vCache" src="./../docs/vCache_Logo_For_Dark_Background.png" width="55%">
  </picture>
</p>


<h3 align="center">
Reliable and Efficient Semantic Prompt Caching
</h3>
<br>

## 🧪 Tests

vCache includes both **unit tests** and **integration tests** to ensure correctness and reliability across its modular components.



### Unit Tests

Unit tests verify the **logic of individual module strategies** (e.g., caching policies, embedding engines, similarity evaluators) in isolation.  
They are designed to be fast, deterministic, and independent of external services.

#### Running Unit Tests

```bash
python -m pytest tests/unit/
```


### Integration Tests

Integration tests validate the **end-to-end behavior** of vCache by checking how components interact (e.g., LLM inference + vector DB + thresholding policy).  
They may involve real API calls and require a valid OpenAI key.

#### Running Integration Tests

1. Create a `.env` file and set your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

2. Run the tests using Poetry:

```bash
poetry run pytest tests/integration
```
