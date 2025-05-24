from typing import List, Optional, Tuple

from vcache.config import vCacheConfig
from vcache.vcache_policy.strategies.static_global_threshold import (
    StaticGlobalThresholdPolicy,
)
from vcache.vcache_policy.vcache_policy import vCachePolicy


class vCache:
    def __init__(
        self,
        config: vCacheConfig = vCacheConfig(),
        policy: vCachePolicy = StaticGlobalThresholdPolicy(),
    ):
        self.vectorq_config = config
        self.vectorq_policy = policy
        self.vectorq_policy.setup(config)

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Infer a response from the cache and return the response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the vCacheConfig if provided.
        Returns
            str - The response to be used by the user
        """
        _, response, _ = self.infer_with_cache_info(prompt, system_prompt)
        return response

    def infer_with_cache_info(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Infer a response from the cache and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the vCacheConfig if provided.
        Returns
            Tuple[bool, str, str] - [is_cache_hit, response, nn_response] (the response is the one supposed to be used by the user, the nn_response is for benchmarking purposes)
        """
        if system_prompt is None:
            system_prompt = self.vectorq_config.system_prompt

        return self.vectorq_policy.process_request(prompt, system_prompt)

    def import_data(self, data: List[str]) -> bool:
        # TODO
        return True

    def flush(self) -> bool:
        # TODO
        return True

    def get_statistics(self) -> str:
        # TODO
        return "No statistics available"
