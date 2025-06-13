from typing import List

from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class NoEvictionPolicy(EvictionPolicy):
    """
    No eviction policy implementation that never evicts items.
    """

    def rank(self, embeddings: List[str]) -> List[int]:
        """
        Rank embeddings with no eviction strategy.

        Args:
            embeddings: The embeddings to rank.

        Returns:
            Empty list since no eviction is performed.
        """
        # TODO
        return []
