#!/usr/bin/env python3
"""
Vector Database Configuration Example

This example demonstrates how to configure different vector databases in vCache.
We show how to use FAISS instead of the default HNSWLib, and configure
different similarity metrics and capacities.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install vCache: pip install -e .
- Install FAISS: pip install faiss-cpu (or faiss-gpu)

Usage:
    python examples/03_vector_database.py
"""

import os
from vcache.main import VCache
from vcache.config import VCacheConfig
from vcache.vcache_core.cache.embedding_store.vector_db import SimilarityMetricType
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import HNSWLibVectorDB
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss import FAISSVectorDB


def test_vector_db(vcache, db_name, prompts):
    """Test a vector database configuration with a set of prompts."""
    print(f"\n=== Testing {db_name} ===")
    
    for i, prompt in enumerate(prompts, 1):
        cache_hit, response, _ = vcache.infer_with_cache_info(prompt)
        print(f"{i}. Prompt: {prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Response: {response[:80]}{'...' if len(response) > 80 else ''}\n")


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== Vector Database Configuration Example ===")

    # Test prompts for vector database comparison
    test_prompts = [
        "How does photosynthesis work?",
        "Explain the process of photosynthesis",  # Similar
        "What is cellular respiration?",          # Related but different
        "How do plants make energy?",             # Similar to first
        "What is quantum computing?",             # Different topic
    ]

    # 1. Default HNSWLib with Cosine Similarity
    print("\n1. HNSWLib Vector Database (Default)")
    print("   - Fast approximate nearest neighbor search")
    print("   - Cosine similarity metric")
    print("   - Good for high-dimensional embeddings")
    
    hnsw_config = VCacheConfig(
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=10000,  # Maximum number of vectors to store
        )
    )
    vcache_hnsw = VCache(config=hnsw_config)
    test_vector_db(vcache_hnsw, "HNSWLib (Cosine)", test_prompts)

    # 2. HNSWLib with Euclidean Distance
    print("\n2. HNSWLib with Euclidean Distance")
    print("   - Same algorithm, different similarity metric")
    print("   - Euclidean distance instead of cosine similarity")
    print("   - May behave differently for normalized embeddings")
    
    hnsw_euclidean_config = VCacheConfig(
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.EUCLIDEAN,
            max_capacity=10000,
        )
    )
    vcache_hnsw_euclidean = VCache(config=hnsw_euclidean_config)
    test_vector_db(vcache_hnsw_euclidean, "HNSWLib (Euclidean)", test_prompts)

    # 3. FAISS Vector Database
    try:
        print("\n3. FAISS Vector Database")
        print("   - Facebook's library for efficient similarity search")
        print("   - Optimized for large-scale vector search")
        print("   - Supports GPU acceleration (if available)")
        
        faiss_config = VCacheConfig(
            vector_db=FAISSVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=50000,  # FAISS can handle larger capacities efficiently
            )
        )
        vcache_faiss = VCache(config=faiss_config)
        test_vector_db(vcache_faiss, "FAISS (Cosine)", test_prompts)
        
    except ImportError:
        print("\n3. FAISS Vector Database")
        print("   ❌ FAISS not installed. Install with: pip install faiss-cpu")
        print("   FAISS provides efficient similarity search for large datasets")

    # 4. Small Capacity Example (for demonstration)
    print("\n4. Limited Capacity Vector Database")
    print("   - Small capacity (5 vectors) to demonstrate eviction")
    print("   - When capacity is exceeded, oldest vectors are removed")
    
    small_config = VCacheConfig(
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=5,  # Very small capacity for demonstration
        )
    )
    vcache_small = VCache(config=small_config)
    
    # Add more prompts than capacity to show eviction
    extended_prompts = test_prompts + [
        "What is machine learning?",
        "How does neural network training work?",
        "What is deep learning?",
    ]
    test_vector_db(vcache_small, "Limited Capacity (5 vectors)", extended_prompts)

    print("=== Vector Database Comparison Complete ===")
    print("Key Considerations:")
    print("- HNSWLib: Fast, memory-efficient, good for most use cases")
    print("- FAISS: Highly optimized, supports GPU, better for large scale")
    print("- Cosine similarity: Good for normalized embeddings (like OpenAI)")
    print("- Euclidean distance: May work better for some embedding types")
    print("- Capacity: Balance between memory usage and cache effectiveness")


if __name__ == "__main__":
    main()