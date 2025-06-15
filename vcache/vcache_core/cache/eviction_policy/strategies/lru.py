from datetime import datetime
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class LRUEvictionPolicy(EvictionPolicy):
    """
    Implements a Least Recently Used (LRU) eviction policy.

    This policy evicts items that have not been accessed for the longest time.
    """

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """
        Updates the last_accessed timestamp of the metadata object to the current time.
        """
        metadata.last_accessed = datetime.now()

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """
        Selects victims for eviction based on the least recently used principle.

        This method sorts the metadata by the last_accessed timestamp in
        ascending order. Items that have `None` for last_accessed (i.e., they
        have never been used as a nearest neighbor) are considered the oldest
        and are prioritized for eviction.

        Args:
            all_metadata: A list of all metadata objects in the cache.

        Returns:
            A list of embedding_ids for the items to be evicted.
        """
        sorted_metadata: List[EmbeddingMetadataObj] = sorted(
            all_metadata,
            key=lambda meta: meta.last_accessed
            if meta.last_accessed is not None
            else datetime.min,
        )

        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        victims: List[int] = [
            meta.embedding_id for meta in sorted_metadata[:num_to_evict]
        ]
        return victims

    def __str__(self) -> str:
        """
        Returns a string representation of the LRUEvictionPolicy instance.

        Returns:
            A string representation of the LRUEvictionPolicy instance.
        """
        return (
            f"LRUEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
