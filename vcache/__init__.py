"""
vCache: Reliable and Efficient Semantic Prompt Caching
"""

# Main vCache classes
from .config import VCacheConfig

# Base classes
# Concrete Inference engines
from .inference_engine import (
    BenchmarkInferenceEngine,
    InferenceEngine,
    LangChainInferenceEngine,
    OpenAIInferenceEngine,
    VLLMInferenceEngine,
)
from .main import VCache
from .vcache_core import (
    Cache,
    EmbeddingEngine,
    EmbeddingMetadataObj,
    EmbeddingMetadataStorage,
    EmbeddingStore,
    EvictionPolicy,
    SimilarityEvaluator,
    Statistics,
)

# Concrete Embedding engines
from .vcache_core.cache.embedding_engine import (
    BenchmarkEmbeddingEngine,
    LangChainEmbeddingEngine,
    OpenAIEmbeddingEngine,
)

# Concrete Embedding metadata storage
from .vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
    LangchainMetadataStorage,
)

# Concrete Vector databases
from .vcache_core.cache.embedding_store.vector_db import (
    ChromaVectorDB,
    FAISSVectorDB,
    HNSWLibVectorDB,
    SimilarityMetricType,
    VectorDB,
)

# Concrete Eviction policies
from .vcache_core.cache.eviction_policy import (
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    MRUEvictionPolicy,
    NoEvictionPolicy,
    SCUEvictionPolicy,
)

# Concrete Similarity evaluators
from .vcache_core.similarity_evaluator import (
    BenchmarkComparisonSimilarityEvaluator,
    EmbeddingComparisonSimilarityEvaluator,
    LLMComparisonSimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)

# Concrete vCache Policies
from .vcache_policy import (
    BenchmarkStaticDecisionPolicy,
    BenchmarkVerifiedGlobalDecisionPolicy,
    BenchmarkVerifiedIIDDecisionPolicy,
    NoCachePolicy,
    SigmoidOnlyDecisionPolicy,
    SigmoidProbabilityDecisionPolicy,
    VCachePolicy,
    VerifiedDecisionPolicy,
)

__all__ = [
    # Main classes
    "VCache",
    "VCacheConfig",
    # Base classes
    "VCachePolicy",
    "InferenceEngine",
    "EmbeddingEngine",
    "VectorDB",
    "SimilarityEvaluator",
    "EvictionPolicy",
    "EmbeddingMetadataStorage",
    "Cache",
    "EmbeddingStore",
    "Statistics",
    # Concrete vCache Policies
    "VerifiedDecisionPolicy",
    "NoCachePolicy",
    "SigmoidProbabilityDecisionPolicy",
    "SigmoidOnlyDecisionPolicy",
    "BenchmarkStaticDecisionPolicy",
    "BenchmarkVerifiedGlobalDecisionPolicy",
    "BenchmarkVerifiedIIDDecisionPolicy",
    # Concrete Inference engines
    "OpenAIInferenceEngine",
    "LangChainInferenceEngine",
    "VLLMInferenceEngine",
    "BenchmarkInferenceEngine",
    # Concrete Embedding engines
    "OpenAIEmbeddingEngine",
    "LangChainEmbeddingEngine",
    "BenchmarkEmbeddingEngine",
    # Concrete Vector databases
    "FAISSVectorDB",
    "HNSWLibVectorDB",
    "ChromaVectorDB",
    "SimilarityMetricType",
    # Concrete Similarity evaluators
    "StringComparisonSimilarityEvaluator",
    "LLMComparisonSimilarityEvaluator",
    "EmbeddingComparisonSimilarityEvaluator",
    "BenchmarkComparisonSimilarityEvaluator",
    # Concrete Eviction policies
    "LRUEvictionPolicy",
    "MRUEvictionPolicy",
    "FIFOEvictionPolicy",
    "NoEvictionPolicy",
    "SCUEvictionPolicy",
    # Concrete Embedding metadata storage
    "InMemoryEmbeddingMetadataStorage",
    "LangchainMetadataStorage",
    "EmbeddingMetadataObj",
]
