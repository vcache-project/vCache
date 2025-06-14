# vCache Usage Examples

This directory contains practical examples demonstrating how to use vCache for semantic prompt caching. Each example is self-contained and showcases different aspects of vCache configuration and usage.

## Prerequisites

Before running the examples, ensure you have:

1. **Installed vCache**:
   ```bash
   pip install -e .
   ```

2. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Optional dependencies** (for specific examples):
   ```bash
   # For FAISS vector database example
   pip install faiss-cpu  # or faiss-gpu for GPU support
   ```

## Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)
**What it demonstrates**: The simplest way to use vCache with default configuration.

**Key concepts**:
- Default vCache setup
- Cache hits vs. cache misses
- Semantic similarity detection

**Run it**:
```bash
python examples/01_basic_usage.py
```

### 2. Custom Caching Policies (`02_custom_policy.py`)
**What it demonstrates**: Different caching policies and their behavior.

**Key concepts**:
- Dynamic Local Threshold Policy (default)
- Static Global Threshold Policy
- Error rate vs. cache hit rate trade-offs
- Policy comparison

**Run it**:
```bash
python examples/02_custom_policy.py
```

### 3. Vector Database Configuration (`03_vector_database.py`)
**What it demonstrates**: Different vector database backends and configurations.

**Key concepts**:
- HNSWLib vs. FAISS vector databases
- Cosine similarity vs. Euclidean distance
- Vector database capacity management
- Performance considerations

**Run it**:
```bash
python examples/03_vector_database.py
```

### 4. System Prompts (`04_system_prompts.py`)
**What it demonstrates**: Using system prompts to provide context and instructions.

**Key concepts**:
- System prompt configuration
- Context-specific caching
- Default vs. override system prompts
- Different response styles

**Run it**:
```bash
python examples/04_system_prompts.py
```

### 5. Advanced Configuration (`05_advanced_configuration.py`)
**What it demonstrates**: Comprehensive configuration for different use cases.

**Key concepts**:
- High-performance setup
- Memory-efficient configuration
- Research/development settings
- Production-ready configuration

**Run it**:
```bash
python examples/05_advanced_configuration.py
```

## Understanding the Output

Each example will show:
- **Cache Hit**: `True` if the response came from cache, `False` if it required LLM inference
- **Response**: The actual response returned by vCache
- **Prompt**: The input prompt being processed

### Cache Behavior
- **First query**: Always a cache miss (no cached responses yet)
- **Similar queries**: Should result in cache hits if semantically similar
- **Different topics**: Usually cache misses unless very similar to cached content

## Configuration Components

vCache is highly modular. Here are the main components you can configure:

### Inference Engines
- `OpenAIInferenceEngine`: Uses OpenAI's API for LLM inference
- `LangChainInferenceEngine`: Integrates with LangChain

### Caching Policies
- `DynamicLocalThresholdPolicy`: Learns per-prompt thresholds (recommended)
- `StaticGlobalThresholdPolicy`: Uses fixed threshold for all prompts
- `DynamicGlobalThresholdPolicy`: Learns global threshold
- `IIDLocalThresholdPolicy`: Independent threshold learning
- `NoCachePolicy`: Disables caching (for testing)

### Vector Databases
- `HNSWLibVectorDB`: Fast approximate nearest neighbor search (default)
- `FAISSVectorDB`: Facebook's optimized similarity search library
- `ChromaVectorDB`: Chroma vector database integration

### Similarity Metrics
- `COSINE`: Cosine similarity (recommended for normalized embeddings)
- `EUCLIDEAN`: Euclidean distance
- `DOT_PRODUCT`: Dot product similarity

## Best Practices

1. **Start Simple**: Begin with the basic usage example and default configuration
2. **Tune Gradually**: Adjust one component at a time to understand its impact
3. **Monitor Performance**: Pay attention to cache hit rates and error rates
4. **Choose Appropriate Capacity**: Balance memory usage with cache effectiveness
5. **Test with Real Data**: Use prompts similar to your actual use case

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```
   Please set OPENAI_API_KEY environment variable
   ```
   **Solution**: Set your OpenAI API key as an environment variable

2. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'faiss'
   ```
   **Solution**: Install the required dependency (`pip install faiss-cpu`)

3. **Low Cache Hit Rates**:
   - Try lowering the threshold in static policies
   - Ensure prompts are actually semantically similar
   - Check if system prompts are consistent

4. **High Error Rates**:
   - Increase the threshold in static policies
   - Decrease the delta parameter in dynamic policies
   - Use more conservative similarity metrics

## Next Steps

After running these examples:

1. **Experiment** with your own prompts and use cases
2. **Benchmark** performance with the provided benchmarking tools
3. **Integrate** vCache into your applications
4. **Monitor** cache performance in production
5. **Contribute** improvements back to the project

For more information, see the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) files.