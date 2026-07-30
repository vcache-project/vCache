import math
from typing import List, Optional, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class GPCAEvictionPolicy(EvictionPolicy):
    """
    Grace-Period Confidence-Cost Aware (GPCA) eviction policy.

    Motivation: SCUEvictionPolicy needs >=6 observations on an item before
    VerifiedDecisionPolicy's logistic-regression estimate is trustworthy
    enough to set `t_prime`; until then `t_prime` is None. SCUEvictionPolicy
    treats a None `t_prime` as an infinite (worst-case) distance -- i.e. an
    item that simply hasn't had a chance to prove itself yet is evicted as
    aggressively as the least useful item in the cache, indistinguishable
    from a genuinely bad match. This punishes expensive, rarely-accessed
    items the hardest, since they're the ones least likely to reach 6 hits
    before eviction pressure hits them.

    GPCA replaces that blind spot with a *grace period*: while an item is
    unconfirmed (`t_prime is None`), it is protected in proportion to its
    generation cost and, secondarily, its access frequency so far -- instead
    of being maximally evictable. Once an item is confirmed (`t_prime` is
    set), protection is driven by how reliable it is (low t_prime = high
    confidence the cached response is being reused correctly) blended with
    frequency, similar in spirit to SCU but no longer discarding cost or
    frequency information the way pure SCU does.

    For every item i, define protection_i (higher = more protected, evicted
    last) as:

        Cold-start (t_prime_i is None):
            protection_i = max(0.2, cost'_i, 0.6 * freq'_i)

        Confirmed (t_prime_i is not None):
            protection_i = clip(0.3 + 0.7 * (1 - t_prime_i) + 0.3 * freq'_i, 0, 1)

    where freq'_i is min-max normalized access frequency (observation count)
    across the current cache contents, t_prime_i is already bounded in
    [0, 1] by VerifiedDecisionPolicy, and cost'_i is min-max normalized
    *log-scaled* generation cost. Log-scaling matters here: real generation
    costs are typically heavy-tailed (a handful of extreme outliers next to
    many ordinary items), and normalizing raw cost directly would squash
    nearly every ordinary item's cost'_i down to ~0 next to a rare extreme
    outlier, making the cost term meaningless for most items -- the same
    issue observed empirically in CostAwareEvictionPolicy's raw min-max.

    Eviction priority is priority_i = 1 - protection_i; items with the
    highest priority (least protected) are evicted first.

    With no cost data and no SCU signal at all, protection floors to 0.2
    for every item, i.e. eviction degenerates towards recency-agnostic
    frequency-based behavior rather than silently becoming LRU -- unlike
    CostAwareEvictionPolicy, GPCA does not fall back to pure LRU, since it
    has no staleness term at all by design (frequency + cost + confidence
    only).
    """

    def __init__(
        self,
        max_size: int,
        watermark: float = 0.95,
        eviction_percentage: float = 0.1,
        cold_start_floor: float = 0.2,
        cold_start_freq_weight: float = 0.6,
        confirmed_base: float = 0.3,
        confirmed_confidence_weight: float = 0.7,
        confirmed_freq_weight: float = 0.3,
    ):
        """
        Args:
            max_size: The absolute maximum number of items the cache can hold.
            watermark: The percentage of `max_size` that triggers eviction.
            eviction_percentage: The percentage of `max_size` to evict.
            cold_start_floor: Minimum protection given to every unconfirmed
                item regardless of cost/frequency, in [0, 1].
            cold_start_freq_weight: How much access frequency (before
                confirmation) contributes to protection during the grace
                period, in [0, 1].
            confirmed_base: Baseline protection given to every confirmed
                item regardless of confidence/frequency, in [0, 1].
            confirmed_confidence_weight: How strongly SCU confidence
                (1 - t_prime) contributes to protection once confirmed.
            confirmed_freq_weight: How strongly access frequency contributes
                to protection once confirmed.
        """
        super().__init__(
            max_size=max_size,
            watermark=watermark,
            eviction_percentage=eviction_percentage,
        )
        self.cold_start_floor = cold_start_floor
        self.cold_start_freq_weight = cold_start_freq_weight
        self.confirmed_base = confirmed_base
        self.confirmed_confidence_weight = confirmed_confidence_weight
        self.confirmed_freq_weight = confirmed_freq_weight

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """No per-access bookkeeping needed beyond what VerifiedDecisionPolicy
        and the cache already maintain (`observations`, `t_prime`, `cost`).

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to update.
        """
        pass

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims using the grace-period cost/confidence/frequency blend.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0 or not all_metadata:
            return []

        costs: List[float] = [
            max(m.cost, 0.0) if m.cost is not None else 0.0 for m in all_metadata
        ]
        log_costs: List[float] = [math.log1p(c) for c in costs]
        freqs: List[float] = [float(len(m.observations)) for m in all_metadata]
        norm_cost: List[float] = self._min_max_normalize(log_costs)
        norm_freq: List[float] = self._min_max_normalize(freqs)

        priorities: List[Tuple[int, float]] = []
        for meta, cost_n, freq_n in zip(all_metadata, norm_cost, norm_freq):
            t_prime: Optional[float] = meta.t_prime
            if t_prime is None:
                protection = max(
                    self.cold_start_floor,
                    cost_n,
                    self.cold_start_freq_weight * freq_n,
                )
            else:
                protection = (
                    self.confirmed_base
                    + self.confirmed_confidence_weight * (1 - t_prime)
                    + self.confirmed_freq_weight * freq_n
                )
                protection = min(1.0, max(0.0, protection))

            priority = 1 - protection
            priorities.append((meta.embedding_id, priority))

        priorities.sort(key=lambda x: x[1], reverse=True)
        return [embedding_id for embedding_id, _ in priorities[:num_to_evict]]

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

    def __str__(self) -> str:
        """Returns a string representation of the GPCAEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"GPCAEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage}, "
            f"cold_start_floor={self.cold_start_floor}, "
            f"cold_start_freq_weight={self.cold_start_freq_weight}, "
            f"confirmed_base={self.confirmed_base}, "
            f"confirmed_confidence_weight={self.confirmed_confidence_weight}, "
            f"confirmed_freq_weight={self.confirmed_freq_weight})"
        )
