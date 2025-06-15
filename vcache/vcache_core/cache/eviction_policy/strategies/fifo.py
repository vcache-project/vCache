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
        """
        This method is not used in the FIFO policy, as eviction is based purely
        on insertion order.
        """
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """
        Selects victims for eviction based on the first-in, first-out principle.

        This method sorts the metadata by the created_at timestamp in ascending
        order to find the oldest items.

        Args:
            all_metadata: A list of all metadata objects in the cache.

        Returns:
            A list of embedding_ids for the items to be evicted.
        """
        sorted_metadata: List[EmbeddingMetadataObj] = sorted(
            all_metadata,
            key=lambda meta: meta.created_at,
        )

        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        victims: List[int] = [
            meta.embedding_id for meta in sorted_metadata[:num_to_evict]
        ]
        return victims

    def __str__(self) -> str:
        """
        Returns a string representation of the FIFOEvictionPolicy instance.

        Returns:
            A string representation of the FIFOEvictionPolicy instance.
        """
        return (
            f"FIFOEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
