from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class NoEvictionPolicy(EvictionPolicy):
    """
    A policy that represents the absence of an eviction strategy.

    This policy never flags the cache as ready for eviction and never selects
    any items to be removed. It is suitable for caches that are not size-limited
    or for testing purposes.
    """

    def __init__(self):
        """Initializes the NoEvictionPolicy."""
        # Intentionally override the parent __init__ to ignore sizing parameters.
        pass

    def ready_to_evict(self, cache) -> bool:
        """This policy is never ready to evict."""
        return False

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Does nothing, as no metadata is needed for this policy."""
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Never selects any victims, always returns an empty list."""
        return []
