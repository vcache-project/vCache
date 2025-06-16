import copy
from typing import List, Optional, Tuple

from vcache.config import VCacheConfig
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_policy.strategies.verified import (
    VerifiedDecisionPolicy,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy


class VCache:
    """
    Main VCache class that provides caching functionality for inference operations.
    """

    def __init__(
        self,
        config: VCacheConfig = VCacheConfig(),
        policy: VCachePolicy = VerifiedDecisionPolicy(delta=0.02),
    ):
        """
        Initialize VCache with configuration and policy.

        Args:
            config: VCache configuration object containing system settings.
            policy: VCache policy for determining cache behavior.
        """
        self.vcache_config: VCacheConfig = config
        self.vcache_policy: VCachePolicy = policy
        self.vcache_policy.setup(config)

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Infer a response from the cache and return the response.

        Args:
            prompt: The prompt to create a response for.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            The response to be used by the user.
        """
        _, response, _ = self.infer_with_cache_info(prompt, system_prompt)
        return response

    def infer_with_cache_info(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        """
        Infer a response from the cache and return the cache hit status, the response, and the nearest neighbor response.

        Args:
            prompt: The prompt to create a response for.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            Tuple containing [is_cache_hit, response, nn_metadata] where response is the one supposed to be used by the user, and nn_metadata is for benchmarking purposes.
        """
        if system_prompt is None:
            system_prompt = self.vcache_config.system_prompt

        if self.vcache_config.eviction_policy.is_evicting():
            response = self.__generate_response(prompt, system_prompt)
            return (
                False,
                response,
                EmbeddingMetadataObj(embedding_id=-1, response=response),
            )

        is_cache_hit, response, nn_metadata = self.vcache_policy.process_request(
            prompt, system_prompt
        )

        if nn_metadata is not None:
            self.vcache_config.eviction_policy.update_eviction_metadata(nn_metadata)

        nn_metadata_copy: Optional[EmbeddingMetadataObj] = (
            copy.deepcopy(nn_metadata) if nn_metadata is not None else None
        )

        if self.vcache_config.eviction_policy.ready_to_evict(self.vcache_policy.cache):
            self.vcache_config.eviction_policy.evict(self.vcache_policy.cache)

        return is_cache_hit, response, nn_metadata_copy

    def __generate_response(self, prompt: str, system_prompt: str) -> str:
        response = self.vcache_policy.inference_engine.create(prompt, system_prompt)
        return response

    def import_data(self, data: List[str]) -> bool:
        """
        Import data into the cache.

        Args:
            data: List of strings to import into the cache.

        Returns:
            True if import was successful.
        """
        # TODO
        return True

    def flush(self) -> bool:
        """
        Flush all data from the cache.

        Returns:
            True if flush was successful.
        """
        # TODO
        return True

    def get_statistics(self) -> str:
        """
        Get statistics about cache performance.

        Returns:
            String containing cache statistics.
        """
        # TODO
        return "No statistics available"
