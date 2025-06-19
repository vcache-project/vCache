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
        """Initializes the embedding store.

        Args:
            vector_db (VectorDB): The vector database for storing embeddings.
            embedding_metadata_storage (EmbeddingMetadataStorage): The storage for
                embedding metadata.
        """
        self.vector_db = vector_db
        self.embedding_metadata_storage = embedding_metadata_storage
        self._add_lock = threading.Lock()
        self._remove_lock = threading.Lock()

    def add_embedding(self, embedding: List[float], response: str, id_set: int) -> int:
        """Adds an embedding and its metadata to the store.

        This operation is thread-safe.

        Args:
            embedding (List[float]): The embedding vector to add.
            response (str): The response associated with the embedding.
            id_set (int): The set identifier for the embedding. This is used in the
                benchmark to identify if the nearest neighbor is from the same set
                (if the cached response is correct or incorrect).

        Returns:
            int: The ID of the added embedding.
        """
        with self._add_lock:
            embedding_id = self.vector_db.add(embedding)
            metadata = EmbeddingMetadataObj(
                embedding_id=embedding_id,
                response=response,
                id_set=id_set,
            )
            self.embedding_metadata_storage.add_metadata(
                embedding_id=embedding_id, metadata=metadata
            )
            return embedding_id

    def remove(self, embedding_id: int) -> int:
        """Removes an embedding and its metadata from the store.

        This operation is thread-safe.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        with self._remove_lock:
            self.embedding_metadata_storage.remove_metadata(embedding_id)
            return self.vector_db.remove(embedding_id)

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Gets k-nearest neighbors for a given embedding.

        Args:
            embedding (List[float]): The embedding to find neighbors for.
            k (int): The number of neighbors to return.

        Returns:
            List[tuple[float, int]]: A list of tuples containing similarity scores
            and embedding IDs.
        """
        return self.vector_db.get_knn(embedding, k)

    def reset(self) -> None:
        """Resets the embedding store to an empty state."""
        self.embedding_metadata_storage.flush()
        return self.vector_db.reset()

    def calculate_storage_consumption(self) -> int:
        """Calculates the storage consumption of the embedding store.

        Returns:
            int: The storage consumption in bytes.
        """
        # TODO: Add metadata logic
        return -1

    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        """Gets metadata for a specific embedding.

        Args:
            embedding_id (int): The ID of the embedding.

        Returns:
            EmbeddingMetadataObj: The metadata object for the embedding.
        """
        return self.embedding_metadata_storage.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, metadata: "EmbeddingMetadataObj"
    ) -> "EmbeddingMetadataObj":
        """Updates metadata for a specific embedding.

        Args:
            embedding_id (int): The ID of the embedding.
            metadata (EmbeddingMetadataObj): The new metadata object.

        Returns:
            EmbeddingMetadataObj: The updated metadata object.
        """
        return self.embedding_metadata_storage.update_metadata(embedding_id, metadata)

    def is_empty(self) -> bool:
        """Checks if the embedding store is empty.

        Returns:
            bool: True if the store is empty, False otherwise.
        """
        return self.vector_db.is_empty()

    def vector_db_size(self) -> int:
        """Gets the number of embeddings in the vector database.

        Returns:
            int: The number of embeddings in the vector database.
        """
        return self.vector_db.size()
