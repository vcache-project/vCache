from typing import List, Tuple

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy


class Cache:
    """Cache that manages prompt embeddings and responses using a vector database and metadata store."""

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
        """Generate an embedding for the prompt and add it to the cache.

        This method obtains an embedding for the prompt via the embedding engine,
        stores it in the vector database, and initializes its metadata with the response.

        Args:
            prompt (str): The prompt to cache.
            response (str): The generated response for the prompt.

        Returns:
            int: The unique ID of the new embedding.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.add_embedding(embedding, response)

    def remove(self, embedding_id: int) -> int:
        """Remove an embedding and its metadata from the cache.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        return self.embedding_store.remove(embedding_id)

    def get_knn(self, prompt: str, k: int) -> List[tuple[float, int]]:
        """Retrieve the k closest embeddings for a prompt.

        This method encodes the prompt to a vector and queries the vector database
        for its k nearest neighbors.

        Args:
            prompt (str): The prompt to query.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: List of (similarity_score, embedding_id) tuples.
        """
        embedding = self.embedding_engine.get_embedding(prompt)
        return self.embedding_store.get_knn(embedding, k)

    def flush(self) -> None:
        """Clear all embeddings and metadata from the cache."""
        self.embedding_store.reset()

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """Get metadata associated with a specific embedding.

        Args:
            embedding_id (int): The ID of the embedding.

        Returns:
            EmbeddingMetadataObj: The metadata for the embedding.
        """
        return self.embedding_store.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, embedding_metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """Update metadata for an existing embedding.

        Args:
            embedding_id (int): The ID of the embedding.
            embedding_metadata (EmbeddingMetadataObj): The new metadata object.

        Returns:
            EmbeddingMetadataObj: The updated metadata object.
        """
        return self.embedding_store.update_metadata(embedding_id, embedding_metadata)

    def add_observation(
        self, embedding_id: int, observation: Tuple[float, int]
    ) -> None:
        """Atomically add an observation to an embedding's metadata.

        Args:
            embedding_id (int): The ID of the embedding.
            observation (Tuple[float, int]): A tuple (similarity_score, label).
        """
        self.embedding_store.add_observation(embedding_id, observation)

    def get_current_capacity(self) -> int:
        """Return the current capacity of the cache.

        Returns:
            int: The number of embeddings currently stored.
        """
        # TODO
        return None

    def is_empty(self) -> bool:
        """Check if the cache has no embeddings.

        Returns:
            bool: True if empty, False otherwise.
        """
        return self.embedding_store.is_empty()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """Retrieve all embedding metadata objects.

        Returns:
            List[EmbeddingMetadataObj]: All metadata objects in the cache.
        """
        return self.embedding_store.embedding_metadata_storage.get_all_embedding_metadata_objects()
