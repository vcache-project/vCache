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
pip install -e ".[benchmarks]"
```



## 📁 Datasets

The benchmark script expects input data in a structured JSON format. Example datasets are included in the `data/` directory.

You can download the official benchmarks from Hugging Face:

- **SemCacheLmArena** (chat-style prompts): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)
- **SemCacheClassification** (structured queries): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkClassification)
- **SemCacheSearchQueries** (real-world browser searches): [Dataset ↗](https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries)



## 🚀 Running Benchmarks

Run the main benchmarking script from the project root:

```bash
python benchmarks/benchmark.py
```

Make sure the dataset paths are correctly set in the benchmark configuration, or adapt the script accordingly.



## 📦 Output

Benchmark results are saved to the `results/` directory and include:
- **Error rate** (absolute and relative)
- **Cache reuse rate**
- **LLM inference latency comparison**
- **Precision**, **recall**, and **accuracy** metrics
- **Cache size** and memory impact over time

These metrics help assess trade-offs between reliability, efficiency, and reuse across semantic caching strategies.