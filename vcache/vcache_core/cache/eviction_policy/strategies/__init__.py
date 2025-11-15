from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "LRUEvictionPolicy",
    "MRUEvictionPolicy",
    "FIFOEvictionPolicy",
    "NoEvictionPolicy",
    "SCUEvictionPolicy",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "LRUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.lru",
    "MRUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.mru",
    "FIFOEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.fifo",
    "NoEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.no_eviction",
    "SCUEvictionPolicy": "vcache.vcache_core.cache.eviction_policy.strategies.scu",
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
    from .fifo import FIFOEvictionPolicy
    from .lru import LRUEvictionPolicy
    from .mru import MRUEvictionPolicy
    from .no_eviction import NoEvictionPolicy
    from .scu import SCUEvictionPolicy
