from typing import List

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import VectorDB
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class Cache:
    """
    Cache class that manages embeddings and their associated responses.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        embedding_engine: EmbeddingEngine,
        eviction_policy: EvictionPolicy,
    ):
        """
        Initialize cache with vector database, engine, and eviction policy.

        Args:
            vector_db: Vector database for managing embeddings and metadata.
            embedding_engine: Engine for generating embeddings from text.
            eviction_policy: Policy for removing items when cache is full.
        """
        self.vector_db = vector_db
        self.embedding_engine = embedding_engine
        self.eviction_policy = eviction_policy

    def add(self, prompt: str, response: str) -> int:
        """
        Compute the embedding for the prompt, add an embedding to the vector database with metadata.

        Args:
            prompt: The prompt to add to the cache.
            response: The response to add to the cache.

        Returns:
            The id of the embedding.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        metadata = EmbeddingMetadataObj(response=response)
        embedding_id = self.vector_db.add(embedding, metadata)
        return embedding_id

    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding from the cache.

        Args:
            embedding_id: The id of the embedding to remove.

        Returns:
            The id of the embedding.
        """
        return self.vector_db.remove(embedding_id)

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
        return self.vector_db.get_knn(embedding, k)

    def flush(self) -> None:
        """
        Flush all data from the cache.
        """
        self.vector_db.reset()

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to get the metadata for.

        Returns:
            The metadata of the embedding.
        """
        return self.vector_db.get_metadata(embedding_id)

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
        return self.vector_db.update_metadata(embedding_id, embedding_metadata)

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
        return self.vector_db.is_empty()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in the cache.

        Returns:
            A list of all the embedding metadata objects in the cache.
        """
        return self.vector_db.get_all_embedding_metadata_objects()
