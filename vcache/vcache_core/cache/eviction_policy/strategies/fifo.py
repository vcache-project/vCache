import heapq
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class FIFOEvictionPolicy(EvictionPolicy):
    """
    Implements a First-In, First-Out (FIFO) eviction policy.

    This policy evicts items in the order they were added to the cache.
    """

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
