#!/usr/bin/env python3
"""
Advanced Configuration Example

This example demonstrates advanced vCache configuration by combining multiple
customizations: custom policies, vector databases, similarity evaluators,
and other components to create a highly optimized caching setup.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install vCache: pip install -e .

Usage:
    python examples/05_advanced_configuration.py
"""

import os
from vcache.main import VCache
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.vcache_core.cache.embedding_engine.strategies.open_ai import OpenAIEmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import InMemoryEmbeddingMetadataStorage
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import HNSWLibVectorDB
from vcache.vcache_core.cache.embedding_store.vector_db import SimilarityMetricType
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import StringComparisonSimilarityEvaluator
from vcache.vcache_policy.strategies.dynamic_local_threshold import DynamicLocalThresholdPolicy


def demonstrate_configuration(vcache, config_name, test_prompts):
    """Demonstrate a specific vCache configuration."""
    print(f"\n=== {config_name} ===")
    
    for i, prompt in enumerate(test_prompts, 1):
        cache_hit, response, _ = vcache.infer_with_cache_info(prompt)
        print(f"{i}. Prompt: {prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}\n")


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== Advanced vCache Configuration Example ===")

    # Test prompts for all configurations
    test_prompts = [
        "Explain quantum entanglement",
        "What is quantum entanglement?",      # Similar
        "How does machine learning work?",
        "Can you explain ML algorithms?",     # Similar
        "What is the meaning of life?",       # Different topic
    ]

    # 1. High-Performance Configuration
    print("\n1. High-Performance Configuration")
    print("   - Optimized for speed and efficiency")
    print("   - Large vector database capacity")
    print("   - Conservative error rate (1%)")
    
    high_perf_config = VCacheConfig(
        # Use OpenAI with specific model for consistency
        inference_engine=OpenAIInferenceEngine(model="gpt-3.5-turbo"),
        embedding_engine=OpenAIEmbeddingEngine(model="text-embedding-3-small"),
        
        # Large capacity vector database with cosine similarity
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=100000,  # Large capacity for high-traffic applications
        ),
        
        # In-memory storage for fast access
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        
        # No eviction policy - keep all cached responses
        eviction_policy=NoEvictionPolicy(),
        
        # String comparison for similarity evaluation
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
        
        # Professional system prompt
        system_prompt="You are a knowledgeable assistant providing accurate, concise responses."
    )
    
    # Conservative policy with low error rate
    high_perf_policy = DynamicLocalThresholdPolicy(delta=0.01)  # 1% max error rate
    vcache_high_perf = VCache(config=high_perf_config, policy=high_perf_policy)
    
    demonstrate_configuration(vcache_high_perf, "High-Performance Setup", test_prompts)

    # 2. Memory-Efficient Configuration
    print("\n2. Memory-Efficient Configuration")
    print("   - Optimized for low memory usage")
    print("   - Smaller vector database capacity")
    print("   - Balanced error rate (3%)")
    
    memory_efficient_config = VCacheConfig(
        # Use smaller, more efficient models
        inference_engine=OpenAIInferenceEngine(model="gpt-3.5-turbo"),
        embedding_engine=OpenAIEmbeddingEngine(model="text-embedding-3-small"),
        
        # Smaller capacity vector database
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=1000,  # Smaller capacity to save memory
        ),
        
        # In-memory storage (could be replaced with disk-based storage)
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        
        # No eviction policy
        eviction_policy=NoEvictionPolicy(),
        
        # String comparison evaluator
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
    )
    
    # More permissive policy for higher cache hit rate
    memory_efficient_policy = DynamicLocalThresholdPolicy(delta=0.03)  # 3% max error rate
    vcache_memory_efficient = VCache(config=memory_efficient_config, policy=memory_efficient_policy)
    
    demonstrate_configuration(vcache_memory_efficient, "Memory-Efficient Setup", test_prompts)

    # 3. Research/Development Configuration
    print("\n3. Research/Development Configuration")
    print("   - Optimized for experimentation")
    print("   - Euclidean distance for different similarity behavior")
    print("   - Higher error tolerance for more cache hits")
    
    research_config = VCacheConfig(
        # Standard OpenAI configuration
        inference_engine=OpenAIInferenceEngine(model="gpt-3.5-turbo"),
        embedding_engine=OpenAIEmbeddingEngine(model="text-embedding-3-small"),
        
        # Use Euclidean distance instead of cosine similarity
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.EUCLIDEAN,
            max_capacity=10000,
        ),
        
        # In-memory storage
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        
        # No eviction policy
        eviction_policy=NoEvictionPolicy(),
        
        # String comparison evaluator
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
        
        # Research-oriented system prompt
        system_prompt="You are a research assistant providing detailed, analytical responses with citations when possible."
    )
    
    # More aggressive policy for research (higher cache hit rate)
    research_policy = DynamicLocalThresholdPolicy(delta=0.05)  # 5% max error rate
    vcache_research = VCache(config=research_config, policy=research_policy)
    
    demonstrate_configuration(vcache_research, "Research/Development Setup", test_prompts)

    # 4. Production-Ready Configuration
    print("\n4. Production-Ready Configuration")
    print("   - Balanced for production workloads")
    print("   - Moderate capacity and error rate")
    print("   - Robust configuration for real-world use")
    
    production_config = VCacheConfig(
        # Production-grade OpenAI configuration
        inference_engine=OpenAIInferenceEngine(
            model="gpt-3.5-turbo",
            # Could add additional parameters like temperature, max_tokens, etc.
        ),
        embedding_engine=OpenAIEmbeddingEngine(model="text-embedding-3-small"),
        
        # Balanced vector database configuration
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=25000,  # Moderate capacity for production
        ),
        
        # In-memory storage (in production, consider persistent storage)
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        
        # No eviction policy (in production, consider LRU or time-based eviction)
        eviction_policy=NoEvictionPolicy(),
        
        # String comparison evaluator
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
        
        # Professional system prompt
        system_prompt="You are a helpful, accurate, and professional assistant."
    )
    
    # Production-appropriate policy
    production_policy = DynamicLocalThresholdPolicy(delta=0.02)  # 2% max error rate
    vcache_production = VCache(config=production_config, policy=production_policy)
    
    demonstrate_configuration(vcache_production, "Production-Ready Setup", test_prompts)

    print("=== Advanced Configuration Complete ===")
    print("\nConfiguration Guidelines:")
    print("• High-Performance: Large capacity, low error rate, fast components")
    print("• Memory-Efficient: Small capacity, higher error tolerance")
    print("• Research: Experimental settings, detailed responses")
    print("• Production: Balanced settings, robust for real-world use")
    print("\nKey Considerations:")
    print("• Vector DB capacity vs. memory usage")
    print("• Error rate vs. cache hit rate trade-off")
    print("• Similarity metric choice affects caching behavior")
    print("• System prompts influence response style and caching")
    print("• Choose configuration based on your specific use case")


if __name__ == "__main__":
    main()