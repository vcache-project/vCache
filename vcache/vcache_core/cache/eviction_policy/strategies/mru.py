import heapq
from datetime import datetime, timezone
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class MRUEvictionPolicy(EvictionPolicy):
    """
    Implements a Most Recently Used (MRU) eviction policy.

    This policy evicts items that have been accessed most recently. This can be
    useful in scenarios where older items are more likely to be re-accessed.
    """

    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Updates the metadata object's last-accessed timestamp.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to update.
        """
        metadata.last_accessed = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims for eviction based on the MRU principle.

        This method efficiently finds the items with the largest `last_accessed`
        timestamps using a heap. Items that have `None` for `last_accessed`
        are treated as the oldest and are not prioritized for eviction.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0:
            return []

        victims_metadata: List[EmbeddingMetadataObj] = heapq.nlargest(
            num_to_evict,
            all_metadata,
            key=lambda meta: meta.last_accessed
            if meta.last_accessed is not None
            else self._MIN_DATETIME,
        )

        victims: List[int] = [meta.embedding_id for meta in victims_metadata]
        return victims

    def __str__(self) -> str:
        """Returns a string representation of the MRUEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"MRUEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
