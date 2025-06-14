from abc import ABC, abstractmethod
from typing import Tuple

from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class EvictionPolicy(ABC):
    """
    Abstract base class for cache eviction policies.
    """

    def __init__(
        self, max_size: int, watermark: float = 0.95, eviction_percentage: float = 0.2
    ):
        """
        Initialize the eviction policy.

        The eviction policy is responsible for evicting embeddings and their corresponding metadata
        from the cache when the cache exceeds the watermark. The watermark is a percentage of the max size.
        For example, if the max size is 1000 and the watermark is 0.95, then the cache starts to evict embeddings
        when it exceeds 950 embeddings. The eviction percentage is the percentage of the cache to evict when the watermark is exceeded.
        For example, if the eviction percentage is 0.2, then 20% of the cache will be evicted when the watermark is exceeded.

        Args:
            max_size: The maximum number of embeddings that can be stored in the cache.
            watermark: The watermark of the cache.
            eviction_percentage: The percentage of the cache to evict when the watermark is exceeded.
        """
        self.max_size = max_size
        self.watermark = watermark
        self.eviction_percentage = eviction_percentage

    def ready_to_evict(self, cache: Cache) -> bool:
        """
        Check if the cache is ready to evict.
        """
        number_of_embeddings_in_cache = cache.vector_db_size()
        return number_of_embeddings_in_cache > self.max_size * self.watermark

    @abstractmethod
    def update_eviction_priority(self, metadata: EmbeddingMetadataObj) -> None:
        """
        Update the relevant fields of the embedding metadata object.

        This function is called when an embedding was used as a nearest neighbor. It updates
        the specific field(s) of the metadata object. They are used to determine the eviction priority.

        Args:
            metadata: The embedding metadata object to update the rank of.
        """
        pass

    @abstractmethod
    def evict_embeddings(self, cache: Cache) -> Tuple[int, int]:
        """
        Evict embeddings from the cache.

        This function is being called when the cache exceeds the watermark. It locks the cache
        to prevent race conditions. When the cache is locked, all incoming requests
        are directly processed by the inference engine. The cache is unlocked when the
        eviction is complete. The number of embeddings to evict is determined by the
        eviction percentage. The embeddings are evicted based on the eviction priority. The
        function returns the number of embeddings evicted and the number of embeddings remaining.

        Args:
            cache: The cache to evict embeddings from.

        Returns:
            The number of embeddings evicted and the number of embeddings remaining.
        """
        pass
