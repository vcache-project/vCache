from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "VCachePolicy",
    "BenchmarkStaticDecisionPolicy",
    "SigmoidProbabilityDecisionPolicy",
    "SigmoidOnlyDecisionPolicy",
    "VerifiedDecisionPolicy",
    "BenchmarkVerifiedGlobalDecisionPolicy",
    "BenchmarkVerifiedIIDDecisionPolicy",
    "NoCachePolicy",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "VCachePolicy": "vcache.vcache_policy.vcache_policy",
    "BenchmarkStaticDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_static",
    "SigmoidProbabilityDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_probability",
    "SigmoidOnlyDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_sigmoid_only",
    "VerifiedDecisionPolicy": "vcache.vcache_policy.strategies.verified",
    "BenchmarkVerifiedGlobalDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_verified_global",
    "BenchmarkVerifiedIIDDecisionPolicy": "vcache.vcache_policy.strategies.benchmark_iid_verified",
    "NoCachePolicy": "vcache.vcache_policy.strategies.no_cache",
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
    from .vcache_policy import VCachePolicy
    from .strategies.benchmark_iid_verified import BenchmarkVerifiedIIDDecisionPolicy
    from .strategies.benchmark_sigmoid_only import SigmoidOnlyDecisionPolicy
    from .strategies.benchmark_sigmoid_probability import (
        SigmoidProbabilityDecisionPolicy,
    )
    from .strategies.benchmark_static import BenchmarkStaticDecisionPolicy
    from .strategies.benchmark_verified_global import (
        BenchmarkVerifiedGlobalDecisionPolicy,
    )
    from .strategies.no_cache import NoCachePolicy
    from .strategies.verified import VerifiedDecisionPolicy
