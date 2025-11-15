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
]

_LAZY_IMPORTS: Dict[str, str] = {
    "Cache": "vcache.vcache_core.cache.cache",
    "EmbeddingStore": "vcache.vcache_core.cache.embedding_store.embedding_store",
    "EmbeddingMetadataObj": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj",
    "EmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.embedding_engine",
    "EmbeddingMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage",
    "EvictionPolicy": "vcache.vcache_core.cache.eviction_policy.eviction_policy",
    "SystemMonitor": "vcache.vcache_core.cache.eviction_policy.system_monitor",
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
    from .cache import Cache
    from .embedding_engine.embedding_engine import EmbeddingEngine
    from .embedding_store import EmbeddingStore
    from .embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
        EmbeddingMetadataObj,
    )
    from .embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
        EmbeddingMetadataStorage,
    )
    from .eviction_policy.eviction_policy import EvictionPolicy
    from .eviction_policy.system_monitor import SystemMonitor
