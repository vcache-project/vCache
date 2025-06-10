from abc import ABC, abstractmethod
from typing import Optional

from vcache.config import VCacheConfig


class VCachePolicy(ABC):
    @abstractmethod
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given config.
        config: VCacheConfig - The config to setup the policy with.
        """
        pass

    @abstractmethod
    def process_request(
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """
        Process a request and either return the cached response or generate a new one with an LLM inference.

        Args:
            prompt: str - The prompt to check for cache hit
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        pass
