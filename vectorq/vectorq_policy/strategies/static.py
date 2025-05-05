from typing import Any, Optional

from typing_extensions import override

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


class StaticThresholdPolicy(VectorQPolicy):
    def __init__(
        self,
        threshold: float = 0.8,
    ):
        self.threshold = threshold
        self.inference_engine = None
        self.cache = None

    @override
    def setup(self, inference_engine: InferenceEngine, cache: Cache) -> None:
        """
        Setup the policy with the given inference engine and cache.
        Args
            inference_engine: InferenceEngine - The inference engine to use for the policy.
            cache: Cache - The cache to use for the policy.
        """
        self.inference_engine = inference_engine
        self.cache = cache

    @override
    def process_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **inference_engine_kwargs: Any,
    ) -> tuple[bool, str, str]:
        """
        Process a request and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to check for cache hit and create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
            **inference_engine_kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response], nn_response will be an empty string if the cache is empty.
        """
        assert self.inference_engine is not None, "Inference engine has not been setup"
        assert self.cache is not None, "Cache has not been setup"

        # Get the nearest neighbor response
        knn = self.cache.get_knn(prompt=prompt, k=1)
        if not knn:
            # No entries in the cache, call inference engine directly
            response = self.inference_engine.infer(
                prompt=prompt,
                system_prompt=system_prompt,
                **inference_engine_kwargs,
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, ""

        # Get the metadata of the nearest neighbor
        similarity_score, embedding_id = knn[0]
        metadata = self.cache.get_metadata(embedding_id=embedding_id)

        # Check if the similarity score is greater than the threshold
        is_cache_hit = similarity_score >= self.threshold
        if is_cache_hit:
            return True, metadata.response, metadata.response
        else:
            response = self.inference_engine.infer(
                prompt=prompt,
                system_prompt=system_prompt,
                **inference_engine_kwargs,
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, metadata.response
