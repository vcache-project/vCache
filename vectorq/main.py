from typing import Any, List, Optional, Tuple

from vectorq.config import VectorQConfig
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore
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
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )
        self.vectorq_policy = policy
        self.vectorq_policy.setup(
            inference_engine=self.inference_engine,
            cache=self.cache,
        )

    def infer_with_cache_info(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **inference_engine_kwargs: Any,
    ) -> Tuple[bool, str, str]:
        """
        Infer a response from the cache and return the cache hit status, the response, and the nearest neighbor response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
            inference_engine_kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            Tuple[bool, str, str] - [is_cache_hit, response, nn_response] (the response is the one supposed to be used by the user, the nn_response is for benchmarking purposes)
        """
        if system_prompt is None:
            system_prompt = self.vectorq_config.system_prompt

        return self.vectorq_policy.process_request(
            prompt=prompt,
            system_prompt=system_prompt,
            **inference_engine_kwargs,
        )

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **inference_engine_kwargs: Any,
    ) -> str:
        """
        Infer a response from the cache and return the response.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
            inference_engine_kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            str - The response to be used by the user
        """
        _, response, _ = self.infer_with_cache_info(
            prompt=prompt,
            system_prompt=system_prompt,
            **inference_engine_kwargs,
        )
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
