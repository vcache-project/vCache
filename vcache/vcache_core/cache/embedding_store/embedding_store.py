import threading
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import VectorDB


class EmbeddingStore:
    """
    Store for managing embeddings and their associated metadata.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        embedding_metadata_storage: EmbeddingMetadataStorage,
    ):
        """
        Initialize embedding store with vector database and metadata storage.

        Args:
            vector_db: Vector database for storing embeddings.
            embedding_metadata_storage: Storage for embedding metadata.
        """
        self.vector_db = vector_db
        self.embedding_metadata_storage = embedding_metadata_storage
        self._add_lock = threading.Lock()
        self._remove_lock = threading.Lock()

    def add_embedding(self, embedding: List[float], response: str) -> int:
        """
        Add an embedding to the vector database and a new metadata object.

        This operation is thread-safe.

        Args:
            embedding: The embedding vector to add.
            response: The response associated with the embedding.

        Returns:
            The ID of the added embedding.
        """
        with self._add_lock:
            embedding_id = self.vector_db.add(embedding)
            metadata = EmbeddingMetadataObj(
                embedding_id=embedding_id,
                response=response,
            )
            self.embedding_metadata_storage.add_metadata(
                embedding_id=embedding_id, metadata=metadata
            )
            return embedding_id

    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding and its metadata from the store.

        This operation is thread-safe.

        Args:
            embedding_id: The ID of the embedding to remove.

        Returns:
            The ID of the removed embedding.
        """
        with self._remove_lock:
            self.embedding_metadata_storage.remove_metadata(embedding_id)
            return self.vector_db.remove(embedding_id)

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """
        Get k-nearest neighbors for the given embedding.

        Args:
            embedding: The embedding to find neighbors for.
            k: The number of neighbors to return.

        Returns:
            List of tuples containing similarity scores and embedding IDs.
        """
        return self.vector_db.get_knn(embedding, k)

    def reset(self) -> None:
        """
        Reset the embedding store to empty state.
        """
        self.embedding_metadata_storage.flush()
        return self.vector_db.reset()

    def calculate_storage_consumption(self) -> int:
        """
        Calculate the storage consumption of the embedding store.

        Returns:
            The storage consumption in bytes.
        """
        # TODO: Add metadata logic
        return -1

    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding.

        Returns:
            The metadata object for the embedding.
        """
        return self.embedding_metadata_storage.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, metadata: "EmbeddingMetadataObj"
    ) -> "EmbeddingMetadataObj":
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding.
            metadata: The new metadata object.

        Returns:
            The updated metadata object.
        """
        return self.embedding_metadata_storage.update_metadata(embedding_id, metadata)

    def is_empty(self) -> bool:
        """
        Check if the embedding store is empty.

        Returns:
            True if the store is empty, False otherwise.
        """
        return self.vector_db.is_empty()

    def vector_db_size(self) -> int:
        """
        Get the number of embeddings in the vector database.

        Returns:
            The number of embeddings in the vector database.
        """
        return self.vector_db.size()
