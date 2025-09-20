import heapq
from datetime import datetime, timezone
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class LRUEvictionPolicy(EvictionPolicy):
    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.1
    ):
        """
        Implements a Least Recently Used (LRU) eviction policy.

        This policy evicts items that have not been accessed for the longest time.
        The eviction process is triggered when the number of items in the cache
        exceeds a "high-watermark" threshold, which is a percentage of the
        absolute `max_size`. Once triggered, the policy will evict a number
        of items equivalent to `eviction_percentage` of the `max_size`.

        Example:
            With `max_size=1000`, `watermark=0.9`, and `eviction_percentage=0.2`,
            eviction starts when the cache size grows beyond 900 items. The
            policy will then remove 200 items (0.2 * 1000).

        Args:
            max_size: The absolute maximum number of items the cache can hold.
            watermark: The percentage of `max_size` that triggers eviction.
            eviction_percentage: The percentage of `max_size` to evict.
        """
        super().__init__(
            max_size=max_size,
            watermark=watermark,
            eviction_percentage=eviction_percentage,
        )

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Updates the metadata object's last-accessed timestamp.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to update.
        """
        metadata.last_accessed: datetime = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims for eviction based on the LRU principle.

        This method efficiently finds the items with the smallest `last_accessed`
        timestamps using a heap. Items that have `None` for `last_accessed`
        (i.e., they have never been used as a nearest neighbor) are considered
        the oldest and are prioritized for eviction.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
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
        """Returns a string representation of the LRUEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"LRUEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
