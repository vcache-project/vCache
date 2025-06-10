from typing import Optional

from typing_extensions import override

from vcache.config import VCacheConfig
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_policy.vcache_policy import VCachePolicy


class StaticGlobalThresholdPolicy(VCachePolicy):
    """
    Policy that uses a static global threshold to determine cache hits.
    """

    def __init__(
        self,
        threshold: float = 0.8,
    ):
        """
        Initialize static global threshold policy.

        Args:
            threshold: The similarity threshold to use for cache hits.
        """
        self.threshold = threshold
        self.inference_engine = None
        self.cache = None

    @override
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given configuration.

        Args:
            config: The VCache configuration to use.
        """
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
        Process a request using static global threshold policy.

        Args:
            prompt: The prompt to check for cache hit.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_response].

        Raises:
            ValueError: If policy has not been setup.
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
