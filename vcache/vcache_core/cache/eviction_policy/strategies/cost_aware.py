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
            (
                now
                - (
                    meta.last_accessed
                    if meta.last_accessed is not None
                    else self._MIN_DATETIME
                )
            ).total_seconds()
            for meta in all_metadata
        ]
        costs: List[float] = [
            meta.cost if meta.cost is not None else 0.0 for meta in all_metadata
        ]

        max_staleness: float = max(staleness) or 1.0
        max_cost: float = max(costs) or 1.0

        priorities: List[Tuple[int, float]] = []
        for meta, stale, cost in zip(all_metadata, staleness, costs):
            normalized_staleness: float = stale / max_staleness
            normalized_cost: float = cost / max_cost
            priority: float = (
                1 - self.cost_weight
            ) * normalized_staleness + self.cost_weight * (1 - normalized_cost)
            priorities.append((meta.embedding_id, priority))

        priorities.sort(key=lambda x: x[1], reverse=True)
        victims: List[int] = [
            embedding_id for embedding_id, _ in priorities[:num_to_evict]
        ]
        return victims

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
