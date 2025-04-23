# Semantic Benchmark

This directory contains benchmarking tools for comparing LLM and embedding models.

## LLM and Embedding Comparison Benchmark

The `llm_embedding_comparison.py` script processes datasets that contain partially filled information about LLM and embedding model outputs. It completes the missing fields by:

1. Generating embeddings with the specified embedding models
2. Generating responses with the specified LLM models
3. Calculating costs and measuring latency

### Input Dataset Format

The input dataset should be a JSON array of objects with the following structure:

```json
[
  {
    "ID": 1,
    "ID_Set": 1,
    "Prompt": "Where did Alex grow up?",
    "Prompt_Embedding_A": null,
    "Prompt_Embedding_B": null,
    "Prompt_Embedding_A_Cost": null,
    "Prompt_Embedding_B_Cost": null,
    "Embedding_Model_A": "text-embedding-3-large",
    "Embedding_Model_B": "text-embedding-3-small",
    "Latency_Embedding_Model_A": null,
    "Latency_Embedding_Model_B": null,
    "Answer_LLM_A": null,
    "Answer_LLM_B": null,
    "Answer_LLM_A_Cost": null,
    "Answer_LLM_B_Cost": null,
    "LLM_A": "4o-mini",
    "LLM_B": "4o",
    "Latency_LLM_A": null,
    "Latency_LLM_B": null
  },
  // More samples...
]
```

Fields that are `null` or missing will be filled in by the benchmark.

### Installation Requirements

This benchmark requires additional dependencies:

```bash
pip install openai tiktoken tqdm
```

### Running the Benchmark

To run the benchmark:

```bash
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input path/to/input.json --output path/to/output.json
```

Command line options:

- `--input`: Path to the input JSON file (required)
- `--output`: Path to the output JSON file (default: `input_filled_TIMESTAMP.json`)
- `--api-key`: OpenAI API key (if not set as environment variable)
- `--max-samples`: Maximum number of samples to process

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

### Output

The benchmark will produce a JSON file with the same structure as the input, but with all fields filled in. It includes:

1. Embeddings from the specified embedding models
2. Responses from the specified LLM models
3. Costs for each embedding and LLM response
4. Latency measurements for each call

### Example

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the benchmark with a subset of samples
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input data/prompts.json --output results/filled_prompts.json --max-samples 10
``` 