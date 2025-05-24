from abc import ABC, abstractmethod
from typing import Optional

from vcache.config import VCacheConfig


class vCachePolicy(ABC):
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
        prompt: str - The prompt to check for cache hit
        system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.
        returns: tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        pass
