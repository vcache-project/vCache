from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "SimilarityEvaluator",
    "StringComparisonSimilarityEvaluator",
    "LLMComparisonSimilarityEvaluator",
    "EmbeddingComparisonSimilarityEvaluator",
    "BenchmarkComparisonSimilarityEvaluator",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "SimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.similarity_evaluator",
    "StringComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.string_comparison",
    "LLMComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.llm_comparison",
    "EmbeddingComparisonSimilarityEvaluator": "vcache.vcache_core.similarity_evaluator.strategies.embedding_comparison",
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
    from .similarity_evaluator import SimilarityEvaluator
    from .strategies.benchmark_comparison import BenchmarkComparisonSimilarityEvaluator
    from .strategies.embedding_comparison import EmbeddingComparisonSimilarityEvaluator
    from .strategies.llm_comparison import LLMComparisonSimilarityEvaluator
    from .strategies.string_comparison import StringComparisonSimilarityEvaluator
