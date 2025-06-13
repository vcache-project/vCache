from typing import List

from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class LRUEvictionPolicy(EvictionPolicy):
    """
    Least Recently Used (LRU) eviction policy implementation.
    """

    def rank(self, embeddings: List[str]) -> List[int]:
        """
        Rank embeddings using LRU strategy.

        Args:
            embeddings: The embeddings to rank.

        Returns:
            The ranked embeddings by LRU priority.
        """
        # TODO
        return []
