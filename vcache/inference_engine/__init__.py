from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "InferenceEngine",
    "LangChainInferenceEngine",
    "OpenAIInferenceEngine",
    "BenchmarkInferenceEngine",
    "VLLMInferenceEngine",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "InferenceEngine": "vcache.inference_engine.inference_engine",
    "LangChainInferenceEngine": "vcache.inference_engine.strategies.lang_chain",
    "OpenAIInferenceEngine": "vcache.inference_engine.strategies.open_ai",
    "BenchmarkInferenceEngine": "vcache.inference_engine.strategies.benchmark",
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
    from .inference_engine import InferenceEngine
    from .strategies.benchmark import BenchmarkInferenceEngine
    from .strategies.lang_chain import LangChainInferenceEngine
    from .strategies.open_ai import OpenAIInferenceEngine
    from .strategies.vllm import VLLMInferenceEngine
