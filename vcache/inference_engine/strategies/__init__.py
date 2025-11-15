from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "BenchmarkInferenceEngine",
    "LangChainInferenceEngine",
    "OpenAIInferenceEngine",
    "VLLMInferenceEngine",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "BenchmarkInferenceEngine": "vcache.inference_engine.strategies.benchmark",
    "LangChainInferenceEngine": "vcache.inference_engine.strategies.lang_chain",
    "OpenAIInferenceEngine": "vcache.inference_engine.strategies.open_ai",
    "VLLMInferenceEngine": "vcache.inference_engine.strategies.vllm",
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
    from .benchmark import BenchmarkInferenceEngine
    from .lang_chain import LangChainInferenceEngine
    from .open_ai import OpenAIInferenceEngine
    from .vllm import VLLMInferenceEngine
