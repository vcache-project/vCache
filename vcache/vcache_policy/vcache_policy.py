from abc import ABC, abstractmethod
from typing import Optional

from vcache.config import VCacheConfig


class VCachePolicy(ABC):
    """Abstract base class for vCache caching policies."""

    @abstractmethod
    def setup(self, config: VCacheConfig):
        """Configure the policy with the necessary components.

        This method is called once to initialize the policy with inference engines,
        cache configurations, and other required components.

        Args:
            config (VCacheConfig): The configuration object for the policy.
        """
        pass

    @abstractmethod
    def process_request(
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """Process a request to decide whether to use a cached response.

        This method determines if a prompt can be served from the cache (a hit)
        or if it requires a new generation from the inference engine (a miss).

        Args:
            prompt (str): The user's prompt.
            system_prompt (str, optional): An optional system prompt to guide the LLM.

        Returns:
            tuple[bool, str, str]: A tuple containing:
                - is_cache_hit (bool): True if the response is from the cache.
                - actual_response (str): The response served (from cache or new).
                - nn_response (str): The nearest neighbor's response if one was found.
        """
        pass
