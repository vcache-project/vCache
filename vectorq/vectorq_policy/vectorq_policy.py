from abc import ABC, abstractmethod
from typing import Any, Optional

from vectorq.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache import Cache


class VectorQPolicy(ABC):
    @abstractmethod
    def setup(self, inference_engine: InferenceEngine, cache: Cache) -> None:
        """
        Setup the policy with the given inference engine and cache.
        Args
            inference_engine: InferenceEngine - The inference engine to use for the policy.
            cache: Cache - The cache to use for the policy.
        """
        pass

    @abstractmethod
    def process_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **inference_engine_kwargs: Any,
    ) -> tuple[bool, str, str]:
        """
        Process a request and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to check for cache hit and create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response.
            **inference_engine_kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        pass
