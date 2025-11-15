from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "EmbeddingMetadataStorage",
    "InMemoryEmbeddingMetadataStorage",
    "LangchainMetadataStorage",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "EmbeddingMetadataStorage": "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage",
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
    from .embedding_metadata_storage import EmbeddingMetadataStorage
    from .strategies.in_memory import InMemoryEmbeddingMetadataStorage
    from .strategies.langchain import LangchainMetadataStorage
