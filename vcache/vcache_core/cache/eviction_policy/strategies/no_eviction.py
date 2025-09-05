from typing import TYPE_CHECKING, List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy

if TYPE_CHECKING:
    from vcache.vcache_core.cache.cache import Cache


class NoEvictionPolicy(EvictionPolicy):
    def __init__(self):
        """
        A policy that represents the absence of an eviction strategy.

        This policy never flags the cache as ready for eviction and never selects
        any items to be removed. It is suitable for caches that are not size-limited
        or for testing purposes.
        """
        # Intentionally override the parent __init__ to ignore sizing parameters.
        pass

    def is_evicting(self) -> bool:
        """This policy is never evicting."""
        return False

    def ready_to_evict(self, cache: "Cache") -> bool:
        """This policy is never ready to evict."""
        return False

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Does nothing, as no metadata is needed for this policy."""
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Never selects any victims, always returns an empty list."""
        return []

    def evict(self, cache: "Cache") -> None:
        """Does nothing, as no eviction is needed for this policy."""
        pass

    def __str__(self) -> str:
        """Returns a string representation of the NoEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return "NoEvictionPolicy()"
