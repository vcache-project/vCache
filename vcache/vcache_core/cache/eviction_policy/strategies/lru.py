from typing import List

from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class LRUEvictionPolicy(EvictionPolicy):
    def rank(self, embeddings: List[str]) -> List[int]:
        # TODO
        return []
