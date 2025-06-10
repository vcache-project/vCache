from typing import Optional

from typing_extensions import override

from vcache.config import VCacheConfig
from vcache.vcache_policy.vcache_policy import VCachePolicy


class NoCachePolicy(VCachePolicy):
    """
    Policy that bypasses caching and always computes fresh responses.
    """

    def __init__(self):
        """
        Initialize no cache policy.
        """
        self.inference_engine = None

    @override
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given configuration.

        Args:
            config: The VCache configuration to use.
        """
        self.inference_engine = config.inference_engine

    @override
    def process_request(
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """
        Process a request without using cache.

        Args:
            prompt: The prompt to process.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_response].

        Raises:
            ValueError: If inference engine has not been setup.
        """
        if self.inference_engine is None:
            raise ValueError("Inference engine has not been setup")

        response = self.inference_engine.create(
            prompt=prompt, system_prompt=system_prompt
        )
        return False, response, ""
