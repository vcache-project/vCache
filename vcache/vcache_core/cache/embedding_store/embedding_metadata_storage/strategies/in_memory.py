import threading
from typing import Dict, List, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class InMemoryEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    """In-memory implementation of embedding metadata storage.

    This class stores embedding metadata in a dictionary and uses locks to ensure
    thread-safe access.

    Attributes:
        metadata_storage (Dict[int, EmbeddingMetadataObj]): The dictionary storing metadata.
    """

    def __init__(self):
        self.metadata_storage: Dict[int, EmbeddingMetadataObj] = {}
        self._store_lock = threading.RLock()
        self._entry_locks: Dict[int, threading.Lock] = {}

    def _get_entry_lock(self, embedding_id: int) -> threading.Lock:
        """Get a lock for a specific embedding ID, creating it if needed.

        Args:
            embedding_id (int): The ID of the embedding.

        Returns:
            threading.Lock: The lock for the given embedding ID.
        """
        with self._store_lock:
            if embedding_id not in self._entry_locks:
                self._entry_locks[embedding_id] = threading.Lock()
            return self._entry_locks[embedding_id]

    def add_metadata(self, embedding_id: int, metadata: EmbeddingMetadataObj) -> int:
        """Add metadata for an embedding to the in-memory store.

        Args:
            embedding_id (int): The ID of the embedding.
            metadata (EmbeddingMetadataObj): The metadata object to associate with the ID.

        Returns:
            int: The ID of the embedding.
        """
        with self._store_lock:
            self.metadata_storage[embedding_id] = metadata
        return embedding_id

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """Retrieve metadata for an embedding from the in-memory store.

        Args:
            embedding_id (int): The ID of the embedding to retrieve.

        Returns:
            EmbeddingMetadataObj: The metadata object for the given ID.

        Raises:
            ValueError: If no metadata is found for the embedding ID.
        """
        with self._store_lock:
            if embedding_id not in self.metadata_storage:
                raise ValueError(
                    f"Embedding metadata for embedding id {embedding_id} not found"
                )
            return self.metadata_storage[embedding_id]

    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """Update metadata for an existing embedding in the in-memory store.

        Args:
            embedding_id (int): The ID of the embedding to update.
            metadata (EmbeddingMetadataObj): The new metadata object.

        Returns:
            EmbeddingMetadataObj: The updated metadata object.

        Raises:
            ValueError: If no metadata is found for the embedding ID.
        """
        with self._store_lock:
            if embedding_id not in self.metadata_storage:
                raise ValueError(
                    f"Embedding metadata for embedding id {embedding_id} not found"
                )
            self.metadata_storage[embedding_id] = metadata
            return metadata

    def add_observation(
        self, embedding_id: int, observation: Tuple[float, int]
    ) -> None:
        """Atomically add an observation to an embedding's metadata.

        This method ensures that appending an observation to the list of
        observations is a thread-safe operation.

        Args:
            embedding_id (int): The ID of the embedding to update.
            observation (Tuple[float, int]): The observation tuple (similarity, label).
        """
        entry_lock = self._get_entry_lock(embedding_id)
        with entry_lock:
            metadata = self.get_metadata(embedding_id)
            metadata.observations.append(observation)
            self.update_metadata(embedding_id, metadata)

    def remove_metadata(self, embedding_id: int) -> bool:
        """Remove metadata for a specific embedding ID.

        Args:
            embedding_id (int): The ID of the embedding metadata to remove.

        Returns:
            bool: True if the metadata was found and removed, False otherwise.
        """
        with self._store_lock:
            if embedding_id in self.metadata_storage:
                del self.metadata_storage[embedding_id]
            # Also remove the associated lock
            if embedding_id in self._entry_locks:
                del self._entry_locks[embedding_id]
            return True
        return False

    def flush(self) -> None:
        """Clear all metadata from the in-memory store."""
        with self._store_lock:
            self.metadata_storage = {}
            self._entry_locks = {}

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """Retrieve all metadata objects from the in-memory store.

        Returns:
            List[EmbeddingMetadataObj]: A list of all metadata objects.
        """
        with self._store_lock:
            return list(self.metadata_storage.values())
