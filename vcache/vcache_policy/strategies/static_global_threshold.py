from typing import Optional

from typing_extensions import override

from vcache.config import VectorQConfig
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_policy.vectorq_policy import VectorQPolicy


class StaticGlobalThresholdPolicy(VectorQPolicy):
    def __init__(
        self,
        threshold: float = 0.8,
    ):
        """
        This policy uses a static threshold to determine if a response is a cache hit.

        Args
            threshold: float - The threshold to use
        """
        self.threshold = threshold
        self.inference_engine = None
        self.cache = None

    @override
    def setup(self, config: VectorQConfig):
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )

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
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        knn = self.cache.get_knn(prompt=prompt, k=1)

        if not knn:
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, ""

        similarity_score, embedding_id = knn[0]
        metadata = self.cache.get_metadata(embedding_id=embedding_id)
        is_cache_hit = similarity_score >= self.threshold
        if is_cache_hit:
            return True, metadata.response, metadata.response
        else:
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, metadata.response
