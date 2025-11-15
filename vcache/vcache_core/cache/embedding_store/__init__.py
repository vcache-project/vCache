from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "VectorDB",
    "EmbeddingStore",
    "SimilarityMetricType",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "VectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
    "EmbeddingStore": "vcache.vcache_core.cache.embedding_store.embedding_store",
    "SimilarityMetricType": "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
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
    from .embedding_store import EmbeddingStore
    from .vector_db.vector_db import SimilarityMetricType, VectorDB
