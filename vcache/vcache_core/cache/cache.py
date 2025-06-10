from typing import List

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class Cache:
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        embedding_engine: EmbeddingEngine,
        eviction_policy: EvictionPolicy,
    ):
        self.embedding_store = embedding_store
        self.embedding_engine = embedding_engine
        self.eviction_policy = eviction_policy

    def add(self, prompt: str, response: str) -> int:
        """
        Generates an embedding for the prompt and adds it to the vector database.
        Initializes the metadata object using the response and stores it in the embedding metadata storage.

        Args:
            prompt: str - The prompt to add to the cache
            response: str - The response to add to the cache
            
        Returns:
            int - The id of the embedding
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        self.embedding_store.add_embedding(embedding, response)

    def remove(self, embedding_id: int) -> int:
        """
        Removes the embedding from the vector database and the embedding metadata storage.

        Args:
            embedding_id: int - The id of the embedding to remove

        Returns:
            int - The id of the embedding
        """
        self.embedding_store.remove(embedding_id)

    def get_knn(self, prompt: str, k: int) -> List[tuple[float, int]]:
        """
        Generates an embedding for the prompt and returns the k-nearest neighbors.

        Args:
            prompt: str - The prompt to get the k-nearest neighbors for
            k: int - The number of nearest neighbors to get

        Returns:
            List[tuple[float, int]] - A list of tuples, each containing a similarity score and an embedding id
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.get_knn(embedding, k)

    def flush(self) -> None:
        """
        Flushes the cache
        """
        self.embedding_store.reset()

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Retrieves the metadata object for the given embedding id.

        Args:
            embedding_id: int - The id of the embedding to get the metadata for

        Returns:
            EmbeddingMetadataObj - The metadata of the embedding
        """
        return self.embedding_store.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, embedding_metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """
        Updates the metadata object for the given embedding id.

        Args:
            embedding_id: int - The id of the embedding to update
            embedding_metadata: EmbeddingMetadataObj - The metadata to update the embedding with

        Returns:
            EmbeddingMetadataObj - The updated metadata of the embedding
        """
        self.embedding_store.update_metadata(embedding_id, embedding_metadata)

    def get_current_capacity(self) -> int:
        """
        returns: int - The current capacity of the cache
        """
        # TODO
        return None

    def is_empty(self) -> bool:
        """
        returns: bool - Whether the cache is empty
        """
        return self.embedding_store.is_empty()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Retrieves all the embedding metadata objects from the embedding metadata storage.

        Returns:
            List["EmbeddingMetadataObj"] - A list of all the embedding metadata objects in the cache
        """
        return self.embedding_store.embedding_metadata_storage.get_all_embedding_metadata_objects()
