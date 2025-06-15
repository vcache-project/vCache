from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.fifo import FIFOEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.mru import MRUEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)

__all__ = [
    "LRUEvictionPolicy",
    "MRUEvictionPolicy",
    "FIFOEvictionPolicy",
    "NoEvictionPolicy",
    "EvictionPolicy",
]
