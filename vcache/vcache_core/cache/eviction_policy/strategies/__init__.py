from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)

__all__ = [
    "LRUEvictionPolicy",
    "NoEvictionPolicy",
]
