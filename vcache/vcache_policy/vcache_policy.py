from abc import ABC, abstractmethod
from typing import Optional

from vcache.config import VCacheConfig


class VCachePolicy(ABC):
    """
    Abstract base class for VCache policies.
    """

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
