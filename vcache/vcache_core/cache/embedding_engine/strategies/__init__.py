from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "BenchmarkEmbeddingEngine",
    "LangChainEmbeddingEngine",
    "OpenAIEmbeddingEngine",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "BenchmarkEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.benchmark",
    "LangChainEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.lang_chain",
    "OpenAIEmbeddingEngine": "vcache.vcache_core.cache.embedding_engine.strategies.open_ai",
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
    from .benchmark import BenchmarkEmbeddingEngine
    from .lang_chain import LangChainEmbeddingEngine
    from .open_ai import OpenAIEmbeddingEngine
