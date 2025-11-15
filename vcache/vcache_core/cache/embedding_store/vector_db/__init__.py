from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "VectorDB",
    "SimilarityMetricType",
    "HNSWLibVectorDB",
    "FAISSVectorDB",
    "ChromaVectorDB",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "VectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
    "SimilarityMetricType": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
    "HNSWLibVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib",
    "FAISSVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss",
    "ChromaVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.chroma",
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
    from .vector_db import SimilarityMetricType, VectorDB
    from .strategies.chroma import ChromaVectorDB
    from .strategies.faiss import FAISSVectorDB
    from .strategies.hnsw_lib import HNSWLibVectorDB
