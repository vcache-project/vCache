from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "EmbeddingComparisonSimilarityEvaluator",
    "LLMComparisonSimilarityEvaluator",
    "StringComparisonSimilarityEvaluator",
    "BenchmarkComparisonSimilarityEvaluator",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "EmbeddingComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.embedding_comparison",
    "LLMComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.llm_comparison",
    "StringComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.string_comparison",
    "BenchmarkComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison",
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
    from .benchmark_comparison import BenchmarkComparisonSimilarityEvaluator
    from .embedding_comparison import EmbeddingComparisonSimilarityEvaluator
    from .llm_comparison import LLMComparisonSimilarityEvaluator
    from .string_comparison import StringComparisonSimilarityEvaluator
