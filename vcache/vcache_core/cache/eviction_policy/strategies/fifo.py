import heapq
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class FIFOEvictionPolicy(EvictionPolicy):
    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.1
    ):
        """
        Implements a First-In, First-Out (FIFO) eviction policy.

        This policy evicts items in the order they were added to the cache.
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
        """This method is not used in the FIFO policy."""
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims for eviction based on the FIFO principle.

        This method efficiently finds the oldest items based on their
        `created_at` timestamp using a heap.

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
            key=lambda meta: meta.created_at,
        )

        victims: List[int] = [meta.embedding_id for meta in victims_metadata]
        return victims

    def __str__(self) -> str:
        """Returns a string representation of the FIFOEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"FIFOEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
