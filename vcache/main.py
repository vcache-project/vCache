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
        """Initializes VCache with configuration and policy.

        Args:
            config (VCacheConfig): Configuration object containing system settings.
            policy (VCachePolicy): Policy for determining cache behavior.
        """
        self.vcache_config: VCacheConfig = config
        self.vcache_policy: VCachePolicy = policy
        self.vcache_policy.setup(config)

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Infers a response from the cache and returns the response.

        Args:
            prompt (str): The prompt to create a response for.
            system_prompt (Optional[str]): The optional system prompt to use
                for the response. Overrides the system prompt in the
                VCacheConfig if provided.

        Returns:
            str: The response to be used by the user.
        """
        _, response, _, _ = self.infer_with_cache_info(prompt, system_prompt)
        return response

    def infer_with_cache_info(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        id_set: int = -1,
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        """Infers a response and returns the cache hit status and metadata.

        Args:
            prompt (str): The prompt to create a response for.
            system_prompt (Optional[str]): The optional system prompt to use
                for the response. Overrides the system prompt in the
                VCacheConfig if provided.
            id_set (int): The set identifier for the embedding. This is used in the
                benchmark to identify if the nearest neighbor is from the same set
                (if the cached response is correct or incorrect).

        Returns:
            Tuple[bool, str, EmbeddingMetadataObj, EmbeddingMetadataObj]: A tuple containing the cache
                hit status, the response, the metadata of the response and nearest neighbor metadata.
        """
        if system_prompt is None:
            system_prompt = self.vcache_config.system_prompt

        if self.vcache_config.eviction_policy.is_evicting():
            response = self.__generate_response(prompt, system_prompt)
            return (
                False,
                response,
                EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
                EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
            )

        is_cache_hit, response, nn_metadata = self.vcache_policy.process_request(
            prompt, system_prompt, id_set
        )

        if nn_metadata is not None:
            self.vcache_config.eviction_policy.update_eviction_metadata(nn_metadata)

        nn_metadata_copy: Optional[EmbeddingMetadataObj] = (
            copy.deepcopy(nn_metadata) if nn_metadata is not None else None
        )

        if self.vcache_config.eviction_policy.ready_to_evict(self.vcache_policy.cache):
            self.vcache_config.eviction_policy.evict(self.vcache_policy.cache)

        if is_cache_hit:
            return is_cache_hit, response, nn_metadata_copy, nn_metadata_copy
        else:
            return (
                is_cache_hit,
                response,
                EmbeddingMetadataObj(embedding_id=-1, response=response, id_set=id_set),
                nn_metadata_copy,
            )

    def __generate_response(self, prompt: str, system_prompt: str) -> str:
        """Generates a new response using the inference engine.

        Args:
            prompt (str): The prompt to generate a response for.
            system_prompt (str): The system prompt to use for generation.

        Returns:
            str: The newly generated response.
        """
        response = self.vcache_policy.inference_engine.create(prompt, system_prompt)
        return response

    def import_data(self, data: List[str]) -> bool:
        """Imports data into the cache.

        Args:
            data (List[str]): A list of strings to import into the cache.

        Returns:
            bool: True if the import was successful.
        """
        # TODO
        return True

    def flush(self) -> bool:
        """Flush all data from the cache.

        Returns:
            bool: True if the flush was successful.
        """
        # TODO
        return True

    def get_statistics(self) -> str:
        """Gets statistics about cache performance.

        Returns:
            str: A string containing cache statistics.
        """
        # TODO
        return "No statistics available"
