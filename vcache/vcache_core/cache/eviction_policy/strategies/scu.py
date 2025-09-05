import math
from typing import List, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy


class SCUEvictionPolicy(EvictionPolicy):
    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.1
    ):
        """
        Implements the Sky Confident Utility (SCU) eviction policy.

        IMPORTANT: This policy can only be used with the VCacheLocal policy.

        This policy uses a Pareto-optimal, distance-from-ideal framework to select
        victims for eviction, balancing an item's generality and the statistical
        confidence in its performance.
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
        """This method is not used in the SCU policy."""
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims for eviction using the SCU model.

        This method orchestrates the victim selection process by:
        1. Checking if enough metadata is available to run the SCU algorithm.
        2. If not, falling back to a LRU-based eviction.
        3. If so, calculating utility scores for all items.
        4. Sorting items by utility and selecting the worst performers.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
        """
        if not all_metadata:
            return []

        eligible_metadata: List[EmbeddingMetadataObj] = [
            meta for meta in all_metadata if meta.t_prime is not None
        ]

        if not eligible_metadata:
            return self._handle_fallback_eviction(all_metadata)

        utilities: List[Tuple[int, float]] = self._calculate_utility_scores(
            all_metadata, eligible_metadata
        )

        utilities.sort(key=lambda x: x[1], reverse=True)

        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        victims: List[int] = [
            embedding_id for embedding_id, _ in utilities[:num_to_evict]
        ]

        return victims

    def _handle_fallback_eviction(
        self, all_metadata: List[EmbeddingMetadataObj]
    ) -> List[int]:
        """Handles eviction by falling back to LRU.

        This is used when no items have been evaluated by the SCU policy.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the oldest items.
        """
        lru_eviction_policy: LRUEvictionPolicy = LRUEvictionPolicy(
            self.max_size, self.watermark, self.eviction_percentage
        )
        return lru_eviction_policy.select_victims(all_metadata)

    def _calculate_utility_scores(
        self,
        all_metadata: List[EmbeddingMetadataObj],
        eligible_metadata: List[EmbeddingMetadataObj],
    ) -> List[Tuple[int, float]]:
        """Calculates a utility score for each metadata object.

        The score is the normalized Euclidean distance from an "ideal point"
        of (t_prime=0, n_obs=max_n_obs). Items not yet evaluated by the
        policy are assigned an infinite distance to mark them as low utility.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata objects
                in the cache.
            eligible_metadata (List[EmbeddingMetadataObj]): A list of metadata objects
                that have valid statistical parameters.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing an
            embedding_id and its calculated utility (distance) score.
        """
        max_n_obs: int = max(
            (len(meta.observations) for meta in eligible_metadata), default=1
        )
        if max_n_obs == 0:
            max_n_obs = 1

        utilities: List[Tuple[int, float]] = []
        for meta in all_metadata:
            if meta.t_prime is None:
                distance: float = float("inf")
            else:
                n_obs_norm: float = (
                    len(meta.observations) / max_n_obs if max_n_obs > 0 else 0.0
                )
                distance: float = math.sqrt(meta.t_prime**2 + (n_obs_norm - 1) ** 2)
            utilities.append((meta.embedding_id, distance))
        return utilities

    def __str__(self) -> str:
        """Returns a string representation of the SCUEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"SCUEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage})"
        )
