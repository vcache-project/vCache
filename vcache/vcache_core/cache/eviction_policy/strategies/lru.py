import heapq
from datetime import datetime, timezone
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

    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """
        Updates the last_accessed timestamp of the metadata object to the current time.
        """
        metadata.last_accessed: datetime = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """
        Selects victims for eviction based on the least recently used principle.

        This method efficiently finds the items with the smallest `last_accessed`
        timestamps using a heap. Items that have `None` for `last_accessed`
        (i.e., they have never been used as a nearest neighbor) are considered
        the oldest and are prioritized for eviction.

        Args:
            all_metadata: A list of all metadata objects in the cache.

        Returns:
            A list of embedding_ids for the items to be evicted.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0:
            return []

        victims_metadata: List[EmbeddingMetadataObj] = heapq.nsmallest(
            num_to_evict,
            all_metadata,
            key=lambda meta: meta.last_accessed
            if meta.last_accessed is not None
            else self._MIN_DATETIME,
        )

        victims: List[int] = [meta.embedding_id for meta in victims_metadata]
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
