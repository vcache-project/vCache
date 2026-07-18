from datetime import datetime, timezone
from typing import List, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class CostAwareEvictionPolicy(EvictionPolicy):
    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __init__(
        self,
        max_size: int,
        watermark: float = 0.95,
        eviction_percentage: float = 0.1,
        cost_weight: float = 0.5,
    ):
        """
        Implements a cost-aware eviction policy.

        This policy evicts items that are both stale (like LRU) and cheap to
        regenerate, while protecting items whose cached response was expensive
        to produce (e.g. a slow LLM call) even if they are accessed less
        frequently. Each item's eviction priority is a weighted blend of its
        normalized staleness and its normalized (inverted) generation cost:

            priority = (1 - cost_weight) * normalized_staleness
                       + cost_weight * (1 - normalized_cost)

        Items with the highest priority are evicted first. Items with unknown
        cost (`None`) are treated as free, i.e. no more protected than under
        plain LRU. With `cost_weight=0.0` this policy is equivalent to LRU.

        The eviction process is triggered when the number of items in the cache
        exceeds a "high-watermark" threshold, which is a percentage of the
        absolute `max_size`. Once triggered, the policy will evict a number
        of items equivalent to `eviction_percentage` of the `max_size`.

        Args:
            max_size: The absolute maximum number of items the cache can hold.
            watermark: The percentage of `max_size` that triggers eviction.
            eviction_percentage: The percentage of `max_size` to evict.
            cost_weight: How strongly generation cost should be weighed against
                staleness, in [0, 1]. Higher values protect expensive items more
                strongly; 0 reduces this policy to plain LRU.
        """
        super().__init__(
            max_size=max_size,
            watermark=watermark,
            eviction_percentage=eviction_percentage,
        )

        if not (0 <= cost_weight <= 1.0):
            cost_weight = 0.5
            self.logger.warning("Cost weight must be in [0,1]. Setting to 0.5.")

        self.cost_weight: float = cost_weight

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Updates the metadata object's last-accessed timestamp.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to update.
        """
        metadata.last_accessed = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims for eviction based on staleness and generation cost.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0 or not all_metadata:
            return []

        now: datetime = datetime.now(timezone.utc)
        staleness: List[float] = [
            self._staleness_seconds(meta, now) for meta in all_metadata
        ]
        costs: List[float] = [
            meta.cost if meta.cost is not None else 0.0 for meta in all_metadata
        ]

        normalized_staleness: List[float] = self._min_max_normalize(staleness)
        normalized_cost: List[float] = self._min_max_normalize(costs)

        priorities: List[Tuple[int, float]] = [
            (
                meta.embedding_id,
                self._compute_priority(stale, cost),
            )
            for meta, stale, cost in zip(
                all_metadata, normalized_staleness, normalized_cost
            )
        ]

        priorities.sort(key=lambda x: x[1], reverse=True)
        victims: List[int] = [
            embedding_id for embedding_id, _ in priorities[:num_to_evict]
        ]
        return victims

    def _staleness_seconds(
        self, metadata: EmbeddingMetadataObj, now: datetime
    ) -> float:
        """Computes how long ago an item was last accessed, in seconds.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to inspect.
            now (datetime): The current time to measure staleness against.

        Returns:
            float: Seconds since `last_accessed`. Items that were never
            accessed are treated as maximally stale.
        """
        last_accessed = (
            metadata.last_accessed
            if metadata.last_accessed is not None
            else self._MIN_DATETIME
        )
        return (now - last_accessed).total_seconds()

    @staticmethod
    def _min_max_normalize(values: List[float]) -> List[float]:
        """Min-max normalizes a list of values to the [0, 1] range.

        Args:
            values (List[float]): The values to normalize.

        Returns:
            List[float]: The normalized values, in the same order. If all
            values are equal (zero range), every value normalizes to 0.0,
            since there is no variation to distinguish them by.
        """
        min_value: float = min(values)
        max_value: float = max(values)
        value_range: float = max_value - min_value
        if value_range == 0:
            return [0.0 for _ in values]
        return [(value - min_value) / value_range for value in values]

    def _compute_priority(
        self, normalized_staleness: float, normalized_cost: float
    ) -> float:
        """Computes an item's eviction priority from its normalized metrics.

        Args:
            normalized_staleness (float): The item's staleness, normalized to [0, 1].
            normalized_cost (float): The item's generation cost, normalized to [0, 1].

        Returns:
            float: The eviction priority. Higher values are evicted first.
        """
        return (1 - self.cost_weight) * normalized_staleness + self.cost_weight * (
            1 - normalized_cost
        )

    def __str__(self) -> str:
        """Returns a string representation of the CostAwareEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"CostAwareEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage}, "
            f"cost_weight={self.cost_weight})"
        )
