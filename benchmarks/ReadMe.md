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

To enable benchmarking capabilities, install vCache with the `benchmarks` extras from the project root:

```bash
pip install -e .[benchmarks]
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

### vCache Datasets

The official benchmark datasets are hosted on Hugging Face and will be downloaded automatically when the script is run:

- **`vCache/SemBenchmarkLmArena`** (chat-style prompts): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)
- **`vCache/SemBenchmarkClassification`** (structured queries): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkClassification)
- **`vCache/SemBenchmarkSearchQueries`** (real-world browser searches): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries)


### Custom Datasets

You can benchmark vCache on your own datasets. The script supports `.csv` and `.parquet` files.

1.  **Place Your Dataset**:
    - Navigate to the directory named `your_datasets` inside the `benchmarks/` directory.
    - Place your custom `.csv` or `.parquet` file inside `benchmarks/your_datasets/`.
    - Your dataset **must** have a column named `prompt`.

2.  **Add to `Dataset` Enum**:
    - Open `benchmarks/benchmark.py`.
    - Add a new entry to the `Dataset` enum. The value should be the relative path from the `benchmarks` directory.

    ```python
    # In benchmarks/benchmark.py
    class Dataset(Enum):
        ...
        # Example for a custom dataset
        MY_AWESOME_DATASET = "your_datasets/my_prompts.csv"
    ```

3.  **Configure the Benchmark Run**:
    - In the `RUN_COMBINATIONS` list in `benchmarks/benchmark.py`, add a new tuple for your benchmark.
    - Use your new `Dataset` enum entry.
    - **Important**: Since custom datasets only contain prompts, you must use live models for inference and embeddings (e.g., `EmbeddingModel.OPENAI_TEXT_EMBEDDING_SMALL`, `LargeLanguageModel.GPT_4_1`). You cannot use the pre-computed models like `GTE` or `E5_LARGE_V2`.
    - For accuracy checking, use a live evaluator like `LLMComparisonSimilarityEvaluator`.

    ```python
    # In benchmarks/benchmark.py
    RUN_COMBINATIONS = [
        (
            EmbeddingModel.OPENAI_TEXT_EMBEDDING_SMALL,
            LargeLanguageModel.GPT_4_1,
            Dataset.MY_AWESOME_DATASET,
            GeneratePlotsOnly.NO,
            LLMComparisonSimilarityEvaluator(
                inference_engine=OpenAIInferenceEngine(model_name="gpt-4o-mini")
            ),
            SCUEvictionPolicy(max_size=2000, watermark=0.99, eviction_percentage=0.1),
            200, # Number of samples to run
        ),
    ]
    ```


## 📦 Output

Benchmark results are saved to the `benchmarks/results/` directory, organized by dataset, embedding model, and LLM. For each run, the output includes:
- **JSON files** containing raw data on cache hits, misses, latency, accuracy metrics, and internal vCache statistics.
- **Plot images (`.png`, `.pdf`)** visualizing key trade-offs, such as cache hit rate vs. accuracy and latency savings.

These metrics help assess the trade-offs between reliability, efficiency, and reuse across different semantic caching strategies.