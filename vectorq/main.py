from typing import List, Optional, Tuple
from typing import List, Tuple

from vectorq.config import VectorQConfig
from vectorq.vectorq_policy.strategies.static import StaticThresholdPolicy
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


class VectorQ:
    """
    VectorQ is a main class that contains the VectorQ semantic prompt caching system.
    """

    def __init__(
        self,
        config: VectorQConfig = VectorQConfig(),
        policy: VectorQPolicy = StaticThresholdPolicy(),
    ):
        self.vectorq_config = config
        self.vectorq_policy = policy
        self.vectorq_policy.setup(config)

    def infer_with_cache_info(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Infer a response from the cache and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        Returns
            Tuple[bool, str, str] - [is_cache_hit, response, nn_response] (the response is the one supposed to be used by the user, the nn_response is for benchmarking purposes)
        """
        if system_prompt is None:
            system_prompt = self.vectorq_config.system_prompt

        return self.vectorq_policy.process_request(prompt, system_prompt)

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Infer a response from the cache and return the response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        Returns
            str - The response to be used by the user
        """
        _, response, _ = self.infer_with_cache_info(prompt, system_prompt)
        return response

    def import_data(self, data: List[str]) -> bool:
        # TODO
        return True

    def flush(self) -> bool:
        # TODO
        return True

    def get_statistics(self) -> str:
        # TODO
        return "No statistics available"
