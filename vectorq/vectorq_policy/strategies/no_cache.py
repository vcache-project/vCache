from typing import Optional

from typing_extensions import override

from vectorq.config import VectorQConfig
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


class NoCachePolicy(VectorQPolicy):
    def __init__(self):
        """
        This policy does not use a cache and will always compute a response.
        """
        self.inference_engine = None

    @override
    def setup(self, config: VectorQConfig):
        self.inference_engine = config.inference_engine

    @override
    def process_request(
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """
        Args
            prompt: str - The prompt to check for cache hit
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        Returns
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        if self.inference_engine is None:
            raise ValueError("Inference engine has not been setup")

        response = self.inference_engine.create(
            prompt=prompt, system_prompt=system_prompt
        )
        return False, response, ""
