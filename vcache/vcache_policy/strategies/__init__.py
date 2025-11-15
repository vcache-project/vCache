from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "NoCachePolicy",
    "VerifiedDecisionPolicy",
    "SigmoidProbabilityDecisionPolicy",
    "SigmoidOnlyDecisionPolicy",
    "BenchmarkStaticDecisionPolicy",
    "BenchmarkVerifiedGlobalDecisionPolicy",
    "BenchmarkVerifiedIIDDecisionPolicy",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "NoCachePolicy": "vcache.vcache_policy.strategies.no_cache",
    "VerifiedDecisionPolicy": "vcache.vcache_policy.strategies.verified",
    "SigmoidProbabilityDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_probability",
    "SigmoidOnlyDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_only",
    "BenchmarkStaticDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_static",
    "BenchmarkVerifiedGlobalDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_verified_global",
    "BenchmarkVerifiedIIDDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_iid_verified",
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
    from .benchmark_iid_verified import BenchmarkVerifiedIIDDecisionPolicy
    from .benchmark_sigmoid_only import SigmoidOnlyDecisionPolicy
    from .benchmark_sigmoid_probability import SigmoidProbabilityDecisionPolicy
    from .benchmark_static import BenchmarkStaticDecisionPolicy
    from .benchmark_verified_global import BenchmarkVerifiedGlobalDecisionPolicy
    from .no_cache import NoCachePolicy
    from .verified import VerifiedDecisionPolicy
