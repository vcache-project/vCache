from typing import Any, Optional

from typing_extensions import override

from vectorq.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache import Cache
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


class NoCachePolicy(VectorQPolicy):
    def __init__(self):
        self.inference_engine = None

    @override
    def setup(self, inference_engine: InferenceEngine, cache: Cache) -> None:
        """
        Setup the policy with the given inference engine and cache.
        Args
            inference_engine: InferenceEngine - The inference engine to use for the policy.
            cache: Cache - The cache to use for the policy.
        """
        self.inference_engine = inference_engine

    @override
    def process_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **inference_engine_kwargs: Any,
    ) -> tuple[bool, str, str]:
        """
        Process a request and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response.
            **inference_engine_kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response], is_cache_hit is always False and nn_response is always an empty string because there is no cache.
        """
        assert self.inference_engine is not None, "Inference engine has not been setup"

        response = self.inference_engine.infer(
            prompt=prompt,
            system_prompt=system_prompt,
            **inference_engine_kwargs,
        )
        return False, response, ""
