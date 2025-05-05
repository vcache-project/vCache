"""
VectorQ: Reliable and Efficient Semantic Prompt Caching
"""

# Main VectorQ classes
from vectorq.config import VectorQConfig

# Inference engines
from vectorq.inference_engine import (
    InferenceEngine,
    LangChainInferenceEngine,
    OpenAIInferenceEngine,
)
from vectorq.main import VectorQ

# Embedding engines
from vectorq.vectorq_core.cache.embedding_engine import (
    EmbeddingEngine,
    LangChainEmbeddingEngine,
    OpenAIEmbeddingEngine,
)

# Embedding metadata storage
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
    InMemoryEmbeddingMetadataStorage,
)

# Vector databases
from vectorq.vectorq_core.cache.embedding_store.vector_db import (
    ChromaVectorDB,
    FAISSVectorDB,
    HNSWLibVectorDB,
    SimilarityMetricType,
    VectorDB,
)

# Eviction policies
from vectorq.vectorq_core.cache.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
)

# Similarity evaluators
from vectorq.vectorq_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)

# VectorQ Policies
from vectorq.vectorq_policy import (
    DynamicGlobalThresholdPolicy,
    DynamicLocalThresholdPolicy,
    NoCachePolicy,
    StaticGlobalThresholdPolicy,
    VectorQPolicy,
)

__all__ = [
    # Main classes
    "VectorQ",
    "VectorQConfig",
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
    # VectorQ Policies
    "VectorQPolicy",
    "DynamicLocalThresholdPolicy",
    "DynamicGlobalThresholdPolicy",
    "StaticGlobalThresholdPolicy",
    "NoCachePolicy",
]
