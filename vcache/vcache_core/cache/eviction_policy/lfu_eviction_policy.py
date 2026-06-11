"""
Least Frequently Used (LFU) Eviction Policy for vCache.

Tracks access frequency per cache entry and evicts the least frequently
used entries when the cache exceeds its watermark threshold.
Ties in frequency are broken by recency (LRU tie-breaking via last_accessed).
"""

from __future__ import annotations

import heapq
from datetime import datetime, timezone
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class LFUEvictionPolicy(EvictionPolicy):
    """
    Least Frequently Used (LFU) eviction policy with LRU tie-breaking.

    Tracks access_count per entry. Evicts lowest-frequency entries first.
    Ties broken by last_accessed (oldest evicted first).

    Args:
        max_size (int): Maximum number of entries before eviction triggers.
        watermark (float): Fraction of max_size that triggers eviction.
        eviction_percentage (float): Fraction of max_size to evict per cycle.
    """

    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.1
    ):
        super().__init__(
            max_size=max_size,
            watermark=watermark,
            eviction_percentage=eviction_percentage,
        )

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """
        Increment access_count and refresh last_accessed on cache hit.

        Args:
            metadata: The metadata object of the accessed cache entry.
        """
        metadata.access_count = getattr(metadata, "access_count", 0) + 1
        metadata.last_accessed = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """
        Select least frequently used entries for eviction.

        Sorts by (access_count ASC, last_accessed ASC) and returns
        eviction_percentage * max_size entries.

        Args:
            all_metadata: All metadata objects in the cache.

        Returns:
            List of embedding_ids to evict.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0 or not all_metadata:
            return []

        victims_metadata: List[EmbeddingMetadataObj] = heapq.nsmallest(
            num_to_evict,
            all_metadata,
            key=lambda m: (
                getattr(m, "access_count", 0),
                m.last_accessed if m.last_accessed is not None else self._MIN_DATETIME,
            ),
        )
        return [m.embedding_id for m in victims_metadata]

    def __str__(self) -> str:
        return (
            f"LFUEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
