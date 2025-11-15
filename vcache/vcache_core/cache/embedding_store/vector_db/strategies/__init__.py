from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = ["ChromaVectorDB", "FAISSVectorDB", "HNSWLibVectorDB"]

_LAZY_IMPORTS: Dict[str, str] = {
    "ChromaVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.chroma",
    "FAISSVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss",
    "HNSWLibVectorDB": "vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib",
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
    from .chroma import ChromaVectorDB
    from .faiss import FAISSVectorDB
    from .hnsw_lib import HNSWLibVectorDB
