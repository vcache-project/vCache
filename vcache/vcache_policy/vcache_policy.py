from abc import ABC, abstractmethod
from typing import Optional

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
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
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """
        Process a request and determine cache hit status.

        Args:
            prompt: The prompt to check for cache hit.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_response].
        """
        pass
