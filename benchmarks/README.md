# VectorQ Benchmarking Framework

This directory contains the benchmarking framework for VectorQ, which allows you to evaluate the performance of VectorQ's semantic caching system.

## Installation

To use the benchmarking tools, install VectorQ with the benchmarks extras:

```bash
pip install -e ".[benchmarks]"
```

## Data

The benchmark script expects data in a specific JSON format. Example datasets are provided in the `data/` directory.

## Running Benchmarks

### VectorQ Performance Benchmark

The main benchmark script can be run from the repository root:

```bash
python benchmarks/benchmark.py
```

### LLM and Embedding Comparison Benchmark

The semantic benchmark module provides tools for comparing different LLM and embedding models:

```bash
python -m benchmarks.semantic_benchmark.llm_embedding_comparison --input path/to/input.json --output path/to/output.json
```

For more details, see the [Semantic Benchmark README](semantic_benchmark/README.md).

## Configuration

The benchmark script supports various configurations that can be modified in `benchmark.py`:

### Embedding Models
```python
EMBEDDING_MODEL_1 = ('embedding_1', 'GteLargeENv1_5', "float32", 1024)
EMBEDDING_MODEL_2 = ('embedding_2', 'E5_Mistral_7B_Instruct', "float16", 4096)
```

### LLM Models
```python
LARGE_LANGUAGE_MODEL_1 = ('response_1', 'Llama_3_8B_Instruct', "float16", None)
LARGE_LANGUAGE_MODEL_2 = ('response_2', 'Llama_3_70B_Instruct', "float16", None)
```

### Threshold Types
```python
# Static thresholds
static_thresholds = np.array([0.74, 0.76, 0.78, 0.8, 0.825, 0.85, 0.875, 0.9, 0.92, 0.94, 0.96])

# Dynamic thresholds
vectorq_rnd_num_ubs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

# Choose threshold type
THRESHOLD_TYPES = ['static', 'dynamic', 'both']
THRESHOLD_TYPE = THRESHOLD_TYPES[2]  # Use both static and dynamic thresholds
```

## Output

Benchmark results are saved to the `results/` directory, including:
- Error rates (relative and absolute)
- Reuse rates
- Inference time comparisons
- Precision, recall, and accuracy metrics
- Cache size analysis

## Offline Plot Generation

Once benchmarks have been completed and results are saved to the `./benchmarks/results` directory, you can use the `plotter_offline.py` script to generate visualizations without re-running the experiments. This is particularly useful when you need to modify plot styles, try different visualization approaches, or create additional charts from existing benchmark data.

```bash
python benchmarks/plotter_offline.py
```

## Available Benchmarks

1. **VectorQ Performance Benchmark** (`benchmark.py`): Evaluates the performance of VectorQ's semantic caching system including cache hit rate, error rate, and latency improvements.

2. **LLM and Embedding Comparison** (`semantic_benchmark/`): Compares different LLM and embedding models by:
   - Generating embeddings with multiple embedding models
   - Generating responses with multiple LLM models
   - Measuring performance metrics (latency, cost, etc.)
   - Processing datasets to fill in missing information
