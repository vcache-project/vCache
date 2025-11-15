from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "Cache",
    "EmbeddingStore",
    "EmbeddingMetadataObj",
    "EmbeddingEngine",
    "EmbeddingMetadataStorage",
    "EvictionPolicy",
    "SystemMonitor",
    "SimilarityEvaluator",
    "StringComparisonSimilarityEvaluator",
    "LLMComparisonSimilarityEvaluator",
    "EmbeddingComparisonSimilarityEvaluator",
    "Statistics",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "Cache": "vcache.vcache_core.cache.cache",
    "EmbeddingStore": "vcache.vcache_core.cache.embedding_store.embedding_store",
    "EmbeddingMetadataObj": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj",
    "EmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.embedding_engine",
    "EmbeddingMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage",
    "EvictionPolicy": "vcache.vcache_core.cache.eviction_policy.eviction_policy",
    "SystemMonitor": "vcache.vcache_core.cache.eviction_policy.system_monitor",
    "SimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.similarity_evaluator",
    "StringComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.string_comparison",
    "LLMComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.llm_comparison",
    "EmbeddingComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.embedding_comparison",
    "Statistics": "vcache.vcache_core.statistics.statistics",
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
    from .cache.cache import Cache
    from .cache.embedding_engine.embedding_engine import EmbeddingEngine
    from .cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
        EmbeddingMetadataObj,
    )
    from .cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
        EmbeddingMetadataStorage,
    )
    from .cache.embedding_store.embedding_store import EmbeddingStore
    from .cache.eviction_policy.eviction_policy import EvictionPolicy
    from .cache.eviction_policy.system_monitor import SystemMonitor
    from .similarity_evaluator.similarity_evaluator import SimilarityEvaluator
    from .similarity_evaluator.strategies.embedding_comparison import (
        EmbeddingComparisonSimilarityEvaluator,
    )
    from .similarity_evaluator.strategies.llm_comparison import (
        LLMComparisonSimilarityEvaluator,
    )
    from .similarity_evaluator.strategies.string_comparison import (
        StringComparisonSimilarityEvaluator,
    )
    from .statistics.statistics import Statistics
