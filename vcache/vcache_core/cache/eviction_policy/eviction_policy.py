from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class EvictionPolicy(ABC):
    """
    Abstract base class defining the interface for cache eviction policies.

    This class provides a standardized framework for implementing custom eviction
    strategies. The core responsibility of a concrete policy is to decide which
    items to remove from the cache when it becomes overfilled.
    """

    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.1
    ):
        """
        Initializes the parameters for the eviction policy.

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
        if not 0 < watermark <= 1.0:
            raise ValueError("Watermark must be between 0 and 1.")
        if not 0 < eviction_percentage <= 1.0:
            raise ValueError("Eviction percentage must be between 0 and 1.")

        self.max_size = max_size
        self.watermark = watermark
        self.eviction_percentage = eviction_percentage
        self.is_evicting_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def shutdown(self):
        """Shuts down the eviction thread pool gracefully."""
        self.executor.shutdown(wait=True)

    def is_evicting(self) -> bool:
        """
        Checks if an eviction process is currently running.

        Returns:
            True if the eviction lock is held, False otherwise.
        """
        return self.is_evicting_lock.locked()

    def ready_to_evict(self, cache: "Cache") -> bool:
        """
        Determines if the cache has breached its high-watermark.

        This method is called after a new item is added to the cache to check
        if an eviction cycle needs to be triggered.

        Args:
            cache: The cache instance to check.

        Returns:
            True if the current cache size exceeds the watermark threshold,
            False otherwise.
        """
        if self.is_evicting():
            return False

        number_of_embeddings_in_cache = cache.vector_db_size()
        watermark_threshold = self.max_size * self.watermark
        return number_of_embeddings_in_cache > watermark_threshold

    @abstractmethod
    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """
        Updates an item's metadata to track its 'freshness' or 'utility'.

        This method is called whenever an item in the cache is accessed (i.e.,
        it was the nearest neighbor for a query). A concrete implementation
        should update the relevant fields on the `metadata` object that are
        used to determine eviction priority (e.g., a timestamp for LRU, a
        hit counter for LFU).

        Args:
            metadata: The metadata object of the accessed cache item.
        """
        pass

    @abstractmethod
    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """
        Selects which items to evict from the cache.

        This is the core logic of the eviction policy. It takes a list of all
        metadata objects currently in the cache and returns a list of
        `embedding_id`s for the items that should be removed.

        The number of victims to select should be based on the
        `eviction_percentage` of the `max_size`.

        Args:
            all_metadata: A list of all metadata objects in the cache.

        Returns:
            A list of `embedding_id`s to be evicted.
        """
        pass

    def evict(self, cache: "Cache"):
        """
        Asynchronously triggers the eviction process if not already running.

        This method submits the eviction task to a background thread and
        returns immediately, preventing the caller from being blocked.

        Args:
            cache: The cache instance to perform eviction on.
        """
        if not self.is_evicting():
            all_metadata = cache.get_all_embedding_metadata_objects()
            victims = self.select_victims(all_metadata)
            self.executor.submit(self._evict_victims, cache, victims)

    def _evict_victims(self, cache: "Cache", victims: List[int]) -> Tuple[int, int]:
        """
        Performs the thread-safe, batch removal of selected items from the cache.

        This method acquires a lock to ensure that the eviction process is
        atomic and does not conflict with other concurrent operations. It
        iterates through the list of victim IDs and removes each one.

        Args:
            cache: The cache instance from which to evict items.
            victims: A list of `embedding_id`s for the items to be evicted.

        Returns:
            A tuple containing:
              - The number of items successfully evicted.
              - The number of items remaining in the cache post-eviction.
        """
        with self.is_evicting_lock:
            if not victims:
                return 0, cache.vector_db_size()

            for victim_id in victims:
                cache.remove(victim_id)

            evicted_count = len(victims)
            remaining_count = cache.vector_db_size()
            return evicted_count, remaining_count
