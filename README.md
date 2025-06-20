<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/vCache_Logo_For_Dark_Background.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/vCache_Logo_For_Light_Background.png">
    <!-- Fallback -->
    <img alt="vCache" src="./docs/vCache_Logo_For_Dark_Background.png" width="55%">
  </picture>
</p>


<h3 align="center">
Reliable and Efficient Semantic Prompt Caching
</h3>
<br>




Semantic caching reduces LLM latency and cost by returning cached model responses for semantically similar prompts (not just exact matches). **vCache** is the first verified semantic cache that **guarantees user-defined error rate bounds**. vCache replaces static thresholds with **online-learned, embedding-specific decision boundaries**—no manual fine-tuning required. This enables reliable cached response reuse across any embedding model or workload.



> [NOTE]
> vCache is currently in active development. Features and APIs may change as we continue to improve the system.




## 🚀 Quick Install

Install vCache in editable mode:

```bash
pip install -e .
```

Then, set your OpenAI key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```
(Note: vCache uses OpenAI by default for both LLM inference and embedding generation, but you can configure any other backend)

Finally, use vCache in your Python code:

```python
from vcache import VCache

vcache = VCache()
response, cache_hit = vcache.create("Is the sky blue?")
```

By default, vCache uses:
- `OpenAIInferenceEngine`
- `OpenAIEmbeddingEngine`
- `HNSWLibVectorDB`
- `InMemoryEmbeddingMetadataStorage`
- `NoEvictionPolicy`
- `StringComparisonSimilarityEvaluator`
- `VerifiedDecisionPolicy` with a maximum failure rate of 2%



## ⚙️ Advanced Configuration

vCache is modular and highly configurable. Below is an example showing how to customize key components:


```python
# 1. Configure the components for vCache
config: VCacheConfig = VCacheConfig(
    inference_engine=OpenAIInferenceEngine(model_name="gpt-4.1-2025-04-14"),
    embedding_engine=OpenAIEmbeddingEngine(model_name="text-embedding-3-small"),
    vector_db=HNSWLibVectorDB(),
    embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
    similarity_evaluator=LLMComparisonSimilarityEvaluator(
        inference_engine=OpenAIInferenceEngine(model_name="gpt-4.1-nano-2025-04-14")
    ),
)

# 2. Choose a caching policy
policy: VCachePolicy = VerifiedDecisionPolicy(delta=0.03)

# 3. Initialize vCache with the configuration and policy
vcache: VCache = VCache(config, policy)

response: str = vcache.infer("Is the sky blue?")
```

You can swap out any component—such as the eviction policy or vector database—for your specific use case.

You can find complete working examples in the [`playground`](playground/) directory:

- [`example_1.py`](playground/example_1.py) - Basic usage with sample data processing
- [`example_2.py`](playground/example_2.py) - Advanced usage with cache hit tracking and timing

These examples demonstrate how to set up vCache with different configurations and process data efficiently.


### Eviction Policy
vCache supports FIFO, LRU, MRU, and a custom SCU eviction policy. See the [Eviction Policy Documentation](vcache/vcache_core/cache/eviction_policy/README.md) for further details.

## 🧠 What Is Semantic Caching?

Semantic caching reduces LLM latency and cost by returning cached model responses for **semantically similar** prompts (not just exact matches)—so you don’t pay for inference cost and latency on repeated questions that have the same answer.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/vCache_architecture.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/vCache_architecture_white.png">
    <!-- Fallback -->
    <img alt="vCache Architecture" src="./docs/vCache_architecture.png" width="50%">
  </picture>
</p>

### Architecture Overview

1. **Embed & Store**  
Each prompt is converted to a fixed-length vector (an “embedding”) and stored in a vector database along with its LLM response.

2. **Nearest-Neighbor Lookup**  
When a new prompt arrives, the cache embeds it and finds its most similar stored prompt using a similarity metric (e.g., cosine similarity).

3. **Similarity Score**  
The system computes a score between 0 and 1 that quantifies how “close” the new prompt is to the retrieved entry.

4. **Decision: Exploit vs. Explore**  
   - **Exploit (cache hit):** If the similarity is above a confidence bound, return the cached response.  
   - **Explore (cache miss):** Otherwise, infer the LLM for a response, add its embedding and answer to the cache, and return it.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/vCache_workflow.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/vCache_workflow_white.png">
    <!-- Fallback -->
    <img alt="vCache Architecture" src="./docs/vCache_workflow.png" width="45%">
  </picture>
</p>

### Why Fixed Thresholds Fall Short
Existing semantic caches rely on a **global static threshold** to decide whether to reuse a cached response (exploit) or invoke the LLM (explore). If the similarity score exceeds this threshold, the cache reuses the response; otherwise, it infers the model. This strategy is fundamentally limited.

- **Uniform threshold, diverse prompts:** A fixed threshold assumes all embeddings are equally distributed—ignoring that similarity is context-dependent.
- **Threshold too low → false positives:** Prompts with low semantic similarity may be incorrectly treated as equivalent, resulting in reused responses that do not match the intended output.
- **Threshold too high → false negatives:** Prompts with semantically equivalent meaning may fail the similarity check, forcing unnecessary LLM inference and reducing cache efficiency.
- **No correctness control:** There is no mechanism to ensure or even estimate how often reused answers will be wrong.

In short, fixed thresholds trade correctness for simplicity and offer no guarantees. Please refer to the [vCache paper](https://arxiv.org/abs/2502.03771) for further details.

### Introducing vCache

vCache overcomes these limitations with two ideas:

- **Per-Prompt Decision Boundary**  
  vCache learns a custom decision boundary for each cached prompt, based on past observations of “how often similarity × actually matched the correct response.”

- **Built-In Error Constraint**  
  You specify a maximum error rate (e.g., 1%). vCache adjusts every per-prompt decision boundary online. The algorithm enforces optimized cache hit rates and does not require offline training or manual fine-tuning.

### Benefits

- **Reliability**  
  Formally bounds the rate of incorrect cache hits to your chosen tolerance.  
- **Performance**  
  Matches or exceeds static-threshold systems in cache hit rate and end-to-end latency.  
- **Simplicity**  
  Plug in any embedding model; vCache learns and adapts automatically at runtime.

  <p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/vCache_core.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/vCache_core_white.png">
    <!-- Fallback -->
    <img alt="vCache Architecture" src="./docs/vCache_core.png" width="50%">
  </picture>
</p>

Please refer to the [vCache paper](https://arxiv.org/abs/2502.03771) for further details.


## 🛠 Developer Guide

For advanced usage and development setup, see the [Developer Guide](ReadMe_Dev.md).



## 📊 Benchmarking vCache

vCache includes a benchmarking framework to evaluate:
- **Cache hit rate**
- **Error rate**
- **Latency improvement**
- **...**

We provide three open benchmarks:
- **SemCacheLmArena** (chat-style prompts) - [Dataset  ↗](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)
- **SemCacheClassification** (classification queries) - [Dataset  ↗](https://huggingface.co/datasets/vCache/SemBenchmarkClassification)
- **SemCacheSearchQueries** (real-world search logs) - [Dataset  ↗](https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries)

See the [Benchmarking Documentation](benchmarks/ReadMe.md) for instructions.

## 📄 Citation

If you use vCache for your research, please cite our [paper](https://arxiv.org/abs/2502.03771).

```bibtex
@article{schroeder2025adaptive,
  title={vCache: Verified Semantic Prompt Caching},
  author={Schroeder, Luis Gaspar and Desai, Aditya and Cuadron, Alejandro and Chu, Kyle and Liu, Shu and Zhao, Mark and Krusche, Stephan and Kemper, Alfons and Zaharia, Matei and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2502.03771},
  year={2025}
}
```