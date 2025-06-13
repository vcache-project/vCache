from typing import List

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class Cache:
    """
    Cache class that manages embeddings and their associated responses.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        embedding_engine: EmbeddingEngine,
        eviction_policy: EvictionPolicy,
    ):
        """
        Initialize cache with embedding store, engine, and eviction policy.

        Args:
            embedding_store: Store for managing embeddings and metadata.
            embedding_engine: Engine for generating embeddings from text.
            eviction_policy: Policy for removing items when cache is full.
        """
        self.embedding_store = embedding_store
        self.embedding_engine = embedding_engine
        self.eviction_policy = eviction_policy

    def add(self, prompt: str, response: str) -> int:
        """
        Compute the embedding for the prompt, add an embedding to the vector database and a new metadata object.

        IMPORTANT: The embedding is computed first and then added to the vector database.
        The metadata object is added last.
        Consider this when implementing asynchronous logic to prevent race conditions.

        Args:
            prompt: The prompt to add to the cache.
            response: The response to add to the cache.

        Returns:
            The id of the embedding.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.add_embedding(embedding, response)

    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding from the cache.

        Args:
            embedding_id: The id of the embedding to remove.

        Returns:
            The id of the embedding.
        """
        return self.embedding_store.remove(embedding_id)

    def get_knn(self, prompt: str, k: int) -> List[tuple[float, int]]:
        """
        Get k-nearest neighbors for a given prompt.

        Args:
            prompt: The prompt to get the k-nearest neighbors for.
            k: The number of nearest neighbors to get.

        Returns:
            A list of tuples, each containing a similarity score and an embedding id.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.get_knn(embedding, k)

    def flush(self) -> None:
        """
        Flush all data from the cache.
        """
        self.embedding_store.reset()

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to get the metadata for.

        Returns:
            The metadata of the embedding.
        """
        return self.embedding_store.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, embedding_metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to update.
            embedding_metadata: The metadata to update the embedding with.

        Returns:
            The updated metadata of the embedding.
        """
        return self.embedding_store.update_metadata(embedding_id, embedding_metadata)

    def get_current_capacity(self) -> int:
        """
        Get the current capacity of the cache.

        Returns:
            The current capacity of the cache.
        """
        # TODO
        return None

    def is_empty(self) -> bool:
        """
        Check if the cache is empty.

        Returns:
            True if the cache is empty, False otherwise.
        """
        return self.embedding_store.is_empty()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in the cache.

        Returns:
            A list of all the embedding metadata objects in the cache.
        """
        return self.embedding_store.embedding_metadata_storage.get_all_embedding_metadata_objects()
