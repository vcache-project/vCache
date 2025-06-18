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
        """Initializes cache with embedding store, engine, and eviction policy.

        Args:
            embedding_store (EmbeddingStore): Store for managing embeddings and metadata.
            embedding_engine (EmbeddingEngine): Engine for generating embeddings from text.
            eviction_policy (EvictionPolicy): Policy for removing items when cache is full.
        """
        self.embedding_store: EmbeddingStore = embedding_store
        self.embedding_engine: EmbeddingEngine = embedding_engine
        self.eviction_policy: EvictionPolicy = eviction_policy

    def add(self, prompt: str, response: str) -> int:
        """Computes and adds an embedding to the vector database and metadata store.

        Note:
            The embedding is computed first, then added to the vector database,
            and the metadata object is added last. This order is important for
            preventing race conditions in asynchronous implementations.

        Args:
            prompt (str): The prompt to add to the cache.
            response (str): The response to associate with the prompt.

        Returns:
            int: The ID of the newly added embedding.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.add_embedding(embedding, response)

    def remove(self, embedding_id: int) -> int:
        """Removes an embedding and its metadata from the cache.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        return self.embedding_store.remove(embedding_id)

    def get_knn(self, prompt: str, k: int) -> List[tuple[float, int]]:
        """Gets k-nearest neighbors for a given prompt.

        Args:
            prompt (str): The prompt to get the k-nearest neighbors for.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            List[tuple[float, int]]: A list of tuples, each containing a
            similarity score and an embedding ID.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.get_knn(embedding, k)

    def flush(self) -> None:
        """
        Removes all data from the cache, resetting the embedding store to an empty state.
        """
        self.embedding_store.reset()

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """Gets metadata for a specific embedding.

        Args:
            embedding_id (int): The ID of the embedding to get the metadata for.

        Returns:
            EmbeddingMetadataObj: The metadata object for the embedding.
        """
        return self.embedding_store.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, embedding_metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """Updates metadata for a specific embedding.

        Args:
            embedding_id (int): The ID of the embedding to update.
            embedding_metadata (EmbeddingMetadataObj): The new metadata to associate
                with the embedding.

        Returns:
            EmbeddingMetadataObj: The updated metadata object.
        """
        return self.embedding_store.update_metadata(embedding_id, embedding_metadata)

    def get_current_capacity(self) -> int:
        """Gets the current capacity of the cache.

        Returns:
            int: The current capacity of the cache.
        """
        # TODO
        return None

    def is_empty(self) -> bool:
        """Checks if the cache is empty.

        Returns:
            bool: True if the cache is empty, False otherwise.
        """
        return self.embedding_store.is_empty()

    def vector_db_size(self) -> int:
        """Gets the number of embeddings in the vector database.

        Returns:
            int: The number of embeddings in the vector database.
        """
        return self.embedding_store.vector_db_size()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """Gets all embedding metadata objects in the cache.

        Returns:
            List[EmbeddingMetadataObj]: A list of all embedding metadata objects.
        """
        return self.embedding_store.embedding_metadata_storage.get_all_embedding_metadata_objects()
