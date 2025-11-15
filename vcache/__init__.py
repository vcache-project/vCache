"""
vCache: Reliable and Efficient Semantic Prompt Caching
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

from .config import VCacheConfig
from .main import VCache


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


_LAZY_IMPORTS: Dict[str, str] = {
    # Base classes
    "VCachePolicy": "vcache.vcache_policy.vcache_policy",
    "InferenceEngine": "vcache.inference_engine.inference_engine",
    "EmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.embedding_engine",
    "VectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
    "SimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.similarity_evaluator",
    "EvictionPolicy": "vcache.vcache_core.cache.eviction_policy.eviction_policy",
    "EmbeddingMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage",
    "Cache": "vcache.vcache_core.cache.cache",
    "EmbeddingStore": "vcache.vcache_core.cache.embedding_store.embedding_store",
    "Statistics": "vcache.vcache_core.statistics.statistics",
    "EmbeddingMetadataObj": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj",
    # Policies
    "VerifiedDecisionPolicy": "vcache.vcache_policy.strategies.verified",
    "NoCachePolicy": "vcache.vcache_policy.strategies.no_cache",
    "SigmoidProbabilityDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_probability",
    "SigmoidOnlyDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_only",
    "BenchmarkStaticDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_static",
    "BenchmarkVerifiedGlobalDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_verified_global",
    "BenchmarkVerifiedIIDDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_iid_verified",
    # Inference engines
    "OpenAIInferenceEngine": "vcache.inference_engine.strategies.open_ai",
    "LangChainInferenceEngine": "vcache.inference_engine.strategies.lang_chain",
    "VLLMInferenceEngine": "vcache.inference_engine.strategies.vllm",
    "BenchmarkInferenceEngine": "vcache.inference_engine.strategies.benchmark",
    # Embedding engines
    "OpenAIEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.open_ai",
    "LangChainEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.lang_chain",
    "BenchmarkEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.benchmark",
    # Vector DBs
    "FAISSVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss",
    "HNSWLibVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib",
    "ChromaVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.chroma",
    "SimilarityMetricType": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
    # Similarity evaluators
    "StringComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.string_comparison",
    "LLMComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.llm_comparison",
    "EmbeddingComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.embedding_comparison",
    "BenchmarkComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison",
    # Eviction policies
    "LRUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.lru",
    "MRUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.mru",
    "FIFOEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.fifo",
    "NoEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.no_eviction",
    "SCUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.scu",
    # Embedding metadata storage
    "InMemoryEmbeddingMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory",
    "LangchainMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.langchain",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module = import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # These imports are only evaluated by type checkers and won't execute at runtime.
    from .vcache_policy.vcache_policy import VCachePolicy
    from .inference_engine.inference_engine import InferenceEngine
    from .vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
    from .vcache_core.cache.embedding_store.vector_db.vector_db import (
        SimilarityMetricType,
        VectorDB,
    )
    from .vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
        EmbeddingMetadataStorage,
    )
    from .vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
        EmbeddingMetadataObj,
    )
    from .vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
    from .vcache_core.cache.cache import Cache
    from .vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
    from .vcache_core.similarity_evaluator.similarity_evaluator import (
        SimilarityEvaluator,
    )
    from .vcache_core.statistics.statistics import Statistics
    from .vcache_policy.strategies.verified import VerifiedDecisionPolicy
    from .vcache_policy.strategies.no_cache import NoCachePolicy
    from .vcache_policy.strategies.benchmark_sigmoid_probability import (
        SigmoidProbabilityDecisionPolicy,
    )
    from .vcache_policy.strategies.benchmark_sigmoid_only import (
        SigmoidOnlyDecisionPolicy,
    )
    from .vcache_policy.strategies.benchmark_static import BenchmarkStaticDecisionPolicy
    from .vcache_policy.strategies.benchmark_verified_global import (
        BenchmarkVerifiedGlobalDecisionPolicy,
    )
    from .vcache_policy.strategies.benchmark_iid_verified import (
        BenchmarkVerifiedIIDDecisionPolicy,
    )
    from .inference_engine.strategies.open_ai import OpenAIInferenceEngine
    from .inference_engine.strategies.lang_chain import LangChainInferenceEngine
    from .inference_engine.strategies.vllm import VLLMInferenceEngine
    from .inference_engine.strategies.benchmark import BenchmarkInferenceEngine
    from .vcache_core.cache.embedding_engine.strategies.open_ai import (
        OpenAIEmbeddingEngine,
    )
    from .vcache_core.cache.embedding_engine.strategies.lang_chain import (
        LangChainEmbeddingEngine,
    )
    from .vcache_core.cache.embedding_engine.strategies.benchmark import (
        BenchmarkEmbeddingEngine,
    )
    from .vcache_core.cache.embedding_store.vector_db.strategies.faiss import (
        FAISSVectorDB,
    )
    from .vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
        HNSWLibVectorDB,
    )
    from .vcache_core.cache.embedding_store.vector_db.strategies.chroma import (
        ChromaVectorDB,
    )
    from .vcache_core.similarity_evaluator.strategies.string_comparison import (
        StringComparisonSimilarityEvaluator,
    )
    from .vcache_core.similarity_evaluator.strategies.llm_comparison import (
        LLMComparisonSimilarityEvaluator,
    )
    from .vcache_core.similarity_evaluator.strategies.embedding_comparison import (
        EmbeddingComparisonSimilarityEvaluator,
    )
    from .vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
        BenchmarkComparisonSimilarityEvaluator,
    )
    from .vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy
    from .vcache_core.cache.eviction_policy.strategies.mru import MRUEvictionPolicy
    from .vcache_core.cache.eviction_policy.strategies.fifo import FIFOEvictionPolicy
    from .vcache_core.cache.eviction_policy.strategies.no_eviction import (
        NoEvictionPolicy as NoEvictionPolicyType,
    )
    from .vcache_core.cache.eviction_policy.strategies.scu import SCUEvictionPolicy
    from .vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
        InMemoryEmbeddingMetadataStorage,
    )
    from .vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.langchain import (
        LangchainMetadataStorage,
    )

    NoEvictionPolicy = NoEvictionPolicyType
