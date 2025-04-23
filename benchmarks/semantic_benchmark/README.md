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
- `--test`: Test mode - only process the first sample (useful for checking API integration)
- `--temperature`: Temperature for LLM response generation (default: 0.7)
- `--top-p`: Top-p (nucleus sampling) value for LLM response generation (default: 1.0)
- `--workers`: Number of parallel workers for processing samples (default: 1)
- `--resume`: Resume processing from where it was stopped previously

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

# Quick test with just the first sample
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input data/prompts.json --output results/test_result.json --test

# Run with custom temperature and top-p values
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input data/prompts.json --output results/creative_results.json --temperature 0.9 --top-p 0.95

# Process samples in parallel (4 workers)
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input data/prompts.json --output results/parallel_results.json --workers 4

# Resume a previously interrupted run
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input data/prompts.json --output results/filled_prompts.json --resume
```

### Generation Parameters Explained

When generating responses from LLMs, two important parameters control the output:

- **Temperature** (0.0 to 2.0): Controls randomness. Lower values (e.g., 0.2) make responses more focused, deterministic, and conservative. Higher values (e.g., 0.8) make output more random and creative.

- **Top-p** (0.0 to 1.0): Controls nucleus sampling. It filters the token selection to the smallest possible set whose cumulative probability exceeds the probability p. A value of 0.9 means considering only tokens comprising the top 90% probability mass.

### Parallelization

The benchmark supports parallel processing of samples, which is especially useful for large datasets:

- Use the `--workers` parameter to specify the number of parallel workers (default: 1).
- Each worker processes a sample independently, allowing multiple API calls to be made concurrently.
- For large datasets (100+ samples), using multiple workers can significantly reduce the total processing time.

Recommended worker settings:
- Small datasets (< 20 samples): 1-2 workers
- Medium datasets (20-100 samples): 2-4 workers
- Large datasets (> 100 samples): 4-8 workers

Note that more workers may lead to API rate limit issues if processing many samples in a short time.

### Robustness Features

The benchmark includes several features to ensure robustness during long-running processes:

#### Incremental Saving

Results are saved to the output file as soon as each sample is processed, rather than waiting until all samples are complete. This ensures:

- If the connection is interrupted or the process is terminated, all processed data is preserved
- You can monitor progress by checking the output file as the benchmark runs
- The benchmark can be safely stopped and restarted without losing results

#### Resumable Processing

The benchmark supports resuming from where it left off if interrupted:

- Use the `--resume` flag to continue processing from where you left off
- The benchmark identifies already processed samples by their ID field
- Only samples that haven't been processed yet will be handled
- This is particularly useful for large datasets where processing might take hours or days

**Important:** For resumable processing to work correctly, each sample in your dataset must have a unique `ID` field.

#### Atomic File Operations

The benchmark uses atomic file operations to prevent corruption:

- Each result is written to a temporary file first
- The temporary file is then renamed to replace the output file
- This ensures the output file is never left in an inconsistent state

#### Error Handling

If an error occurs while processing a sample:

- The error is logged
- The process continues with the next sample
- The failed sample is preserved in its original form 