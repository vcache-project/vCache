from typing import Any, Dict, List, Optional

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class InMemoryEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    """
    In-memory implementation of embedding metadata storage.
    """

    def __init__(self):
        """
        Initialize in-memory embedding metadata storage.
        """
        self.metadata_storage: Dict[int, "EmbeddingMetadataObj"] = {}

    def add_metadata(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to add metadata for.
            metadata: The metadata to add.

        Returns:
            The embedding ID.
        """
        self.metadata_storage[embedding_id] = metadata
        return embedding_id

    def get_metadata(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to get metadata for.

        Returns:
            The metadata for the embedding.

        Raises:
            ValueError: If embedding metadata is not found.
        """
        if embedding_id not in self.metadata_storage:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        else:
            return self.metadata_storage[embedding_id]

    def update_metadata(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to update metadata for.
            metadata: The new metadata to set.

        Returns:
            The updated metadata.

        Raises:
            ValueError: If embedding metadata is not found.
        """
        if embedding_id not in self.metadata_storage:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        else:
            self.metadata_storage[embedding_id] = metadata
            return metadata

    def remove_metadata(self, embedding_id: int) -> bool:
        """
        Remove metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to remove metadata for.

        Returns:
            True if metadata was removed, False if not found.
        """
        if embedding_id in self.metadata_storage:
            del self.metadata_storage[embedding_id]
            return True
        return False

    def flush(self) -> None:
        """
        Flush all metadata from storage.
        """
        self.metadata_storage = {}

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in storage.

        Returns:
            A list of all embedding metadata objects.
        """
        return list(self.metadata_storage.values())
