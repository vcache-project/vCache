from abc import ABC, abstractmethod
from typing import Optional, Tuple

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class VCachePolicy(ABC):
    """
    Abstract base class for VCache policies.
    """

    def __init__(self):
        """
        Initialize the VCache policy.
        """
        self.inference_engine: Optional[InferenceEngine] = None
        self.cache: Optional[Cache] = None
        self.eviction_policy: Optional[EvictionPolicy] = None

    @abstractmethod
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given config.

        Args:
            config: The config to setup the policy with.
        """
        pass

    @abstractmethod
    def process_request(
        self, prompt: str, system_prompt: Optional[str], id_set: int
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        """
        Process a request and determine cache hit status.

        Args:
            prompt: The prompt to check for cache hit.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.
            id_set: The set identifier for the embedding. This is used in the
                benchmark to identify if the nearest neighbor is from the same set
                (if the cached response is correct or incorrect).

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_metadata_object].
        """
        pass
