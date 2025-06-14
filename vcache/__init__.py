"""
vCache: Reliable and Efficient Semantic Prompt Caching
"""

# Main vCache classes
from vcache.config import VCacheConfig

# Inference engines
from vcache.inference_engine import (
    InferenceEngine,
    LangChainInferenceEngine,
    OpenAIInferenceEngine,
)
from vcache.main import VCache

# Embedding engines
from vcache.vcache_core.cache.embedding_engine import (
    EmbeddingEngine,
    LangChainEmbeddingEngine,
    OpenAIEmbeddingEngine,
)

# Embedding metadata storage
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
    InMemoryEmbeddingMetadataStorage,
)

# Vector databases
from vcache.vcache_core.cache.embedding_store.vector_db import (
    ChromaVectorDB,
    FAISSVectorDB,
    HNSWLibVectorDB,
    SimilarityMetricType,
    VectorDB,
)

# Eviction policies
from vcache.vcache_core.cache.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
)

# Similarity evaluators
from vcache.vcache_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)

# vCache Policies
from vcache.vcache_policy import (
    BenchmarkStaticDecisionPolicy,
    BenchmarkVerifiedGlobalDecisionPolicy,
    BenchmarkVerifiedIIDDecisionPolicy,
    NoCachePolicy,
    VCachePolicy,
    VerifiedDecisionPolicy,
)

__all__ = [
    # Main classes
    "VCache",
    "VCacheConfig",
    # Inference engines
    "InferenceEngine",
    "OpenAIInferenceEngine",
    "LangChainInferenceEngine",
    # Embedding engines
    "EmbeddingEngine",
    "OpenAIEmbeddingEngine",
    "LangChainEmbeddingEngine",
    # Vector databases
    "VectorDB",
    "FAISSVectorDB",
    "HNSWLibVectorDB",
    "ChromaVectorDB",
    "SimilarityMetricType",
    # Similarity evaluators
    "SimilarityEvaluator",
    "StringComparisonSimilarityEvaluator",
    # Eviction policies
    "EvictionPolicy",
    "LRUEvictionPolicy",
    # Embedding metadata storage
    "EmbeddingMetadataStorage",
    "InMemoryEmbeddingMetadataStorage",
    # vCache Policies
    "VCachePolicy",
    "VerifiedDecisionPolicy",
    "BenchmarkVerifiedGlobalDecisionPolicy",
    "BenchmarkStaticDecisionPolicy",
    "NoCachePolicy",
    "BenchmarkVerifiedIIDDecisionPolicy",
]
