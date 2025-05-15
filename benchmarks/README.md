# vCache Benchmarking Framework

This directory contains the benchmarking framework for vCache, which allows you to evaluate the performance of vCache's semantic caching system.

## Installation

To use the benchmarking tools, install vCache with the benchmarks extras:

```bash
pip install -e ".[benchmarks]"
```

## Data

The benchmark script expects data in a specific JSON format. Example datasets are provided in the `data/` directory.

## Running Benchmarks

The main benchmark script can be run from the repository root:

```bash
python benchmarks/benchmark.py
```

## Output

Benchmark results are saved to the `results/` directory, including:
- Error rates (relative and absolute)
- Reuse rates
- Inference time comparisons
- Precision, recall, and accuracy metrics
- Cache size analysis
