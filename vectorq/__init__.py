"""
VectorQ: Reliable and Efficient Semantic Prompt Caching
"""

# Main VectorQ classes
from vectorq.main import VectorQ, VectorQBenchmark
from vectorq.config import VectorQConfig

# Inference engines
from vectorq.inference_engine import (
    InferenceEngine,
    OpenAIInferenceEngine,
    LangChainInferenceEngine,
    DummyInferenceEngine
)

# Embedding engines
from vectorq.vectorq_core.cache.embedding_engine import (
    EmbeddingEngine,
    OpenAIEmbeddingEngine,
    LangChainEmbeddingEngine
)

# Vector databases
from vectorq.vectorq_core.cache.embedding_store.vector_db import (
    VectorDB,
    FAISSVectorDB,
    HNSWLibVectorDB,
    ChromaVectorDB,
    SimilarityMetricType
)

# Similarity evaluators
from vectorq.vectorq_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator
)

# Eviction policies
from vectorq.vectorq_core.cache.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
)

# Embedding metadata storage
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
    InMemoryEmbeddingMetadataStorage
)

__all__ = [
    # Main classes
    'VectorQ',
    'VectorQBenchmark',
    'VectorQConfig',
    
    # Inference engines
    'InferenceEngine',
    'OpenAIInferenceEngine',
    'LangChainInferenceEngine',
    'DummyInferenceEngine',
    
    # Embedding engines
    'EmbeddingEngine',
    'OpenAIEmbeddingEngine',
    'LangChainEmbeddingEngine',
    
    # Vector databases
    'VectorDB',
    'FAISSVectorDB',
    'HNSWLibVectorDB',
    'ChromaVectorDB',
    'SimilarityMetricType',
    
    # Similarity evaluators
    'SimilarityEvaluator',
    'StringComparisonSimilarityEvaluator',
    
    # Eviction policies
    'EvictionPolicy',
    'LRUEvictionPolicy', 
    'NoEvictionPolicy',
    
    # Embedding metadata storage
    'EmbeddingMetadataStorage',
    'InMemoryEmbeddingMetadataStorage'
]

