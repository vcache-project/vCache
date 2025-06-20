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



This directory provides the official benchmarking tools for evaluating the performance of **vCache** under various real-world and synthetic workloads.



## ⚙️ Installation

To enable benchmarking capabilities, install vCache with the `benchmarks` extras:

```bash
pip install -e .
```


## 🚀 Running Benchmarks

Run the main benchmarking script from the project root:

```bash
python benchmarks/benchmark.py
```

The script will automatically download the required datasets from Hugging Face based on the configurations in `RUN_COMBINATIONS`.


## ⚙️ Custom Configuration

The primary configuration is done by modifying the global variables in the `benchmarks/benchmark.py` script. This script is designed to benchmark the performance of vCache against several baselines by evaluating cache hit rates, accuracy, latency, and other metrics.

### Key Configuration Variables:

1.  **`RUN_COMBINATIONS`**: This is the most important setting. It's a list of tuples, where each tuple defines a complete benchmark scenario to run. Each tuple contains:
    - `EmbeddingModel`: The embedding model to use (e.g., `EmbeddingModel.GTE`).
    - `LargeLanguageModel`: The large language model to use (e.g., `LargeLanguageModel.GPT_4O_MINI`).
    - `Dataset`: The dataset for the benchmark, specified by its Hugging Face repository ID.
    - `GeneratePlotsOnly`: Set to `GeneratePlotsOnly.YES` to skip running the benchmark and only regenerate plots from existing results.
    - `SimilarityEvaluator`: The strategy for comparing semantic similarity.
    - `EvictionPolicy`: The cache eviction policy to use.

2.  **`BASELINES_TO_RUN`**: A list to specify which caching strategies to evaluate (e.g., `VCacheLocal`, `GPTCache`, `BerkeleyEmbedding`). Every baseline is run for every combination defined in `RUN_COMBINATIONS`.

3.  **`STATIC_THRESHOLDS`**: A list of similarity thresholds for static policies like `GPTCache` and `BerkeleyEmbedding`. The benchmark will run once for each threshold.

4.  **`DELTAS`**: A list of `delta` values for dynamic policies like `vCache`. The benchmark will run once for each delta.

Refer to the docstring in `benchmarks/benchmark.py` for more details on other configuration options like `CONFIDENCE_INTERVALS_ITERATIONS`, `KEEP_SPLIT`, and `MAX_VECTOR_DB_CAPACITY`.



## 📁 Datasets

The official benchmark datasets are hosted on Hugging Face and will be downloaded automatically when the script is run:

- **`vCache/SemBenchmarkLmArena`** (chat-style prompts): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)
- **`vCache/SemBenchmarkClassification`** (structured queries): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkClassification)
- **`vCache/SemBenchmarkSearchQueries`** (real-world browser searches): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries)


## 📦 Output

Benchmark results are saved to the `benchmarks/results/` directory, organized by dataset, embedding model, and LLM. For each run, the output includes:
- **JSON files** containing raw data on cache hits, misses, latency, accuracy metrics, and internal vCache statistics.
- **Plot images (`.png`, `.pdf`)** visualizing key trade-offs, such as cache hit rate vs. accuracy and latency savings.

These metrics help assess the trade-offs between reliability, efficiency, and reuse across different semantic caching strategies.