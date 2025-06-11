from abc import ABC, abstractmethod
from typing import List, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class EmbeddingMetadataStorage(ABC):
    """Abstract base class for embedding metadata storage."""

    @abstractmethod
    def add_metadata(self, embedding_id: int, metadata: EmbeddingMetadataObj) -> int:
        """Add metadata entry for an embedding.

        Args:
            embedding_id (int): The ID of the embedding.
            metadata (EmbeddingMetadataObj): The metadata to add.

        Returns:
            int: The ID of the embedding.
        """
        pass

    @abstractmethod
    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """Retrieve metadata for an embedding.

        Args:
            embedding_id (int): The ID of the embedding.

        Returns:
            EmbeddingMetadataObj: The metadata for the embedding.
        """
        pass

    @abstractmethod
    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """Update metadata for an existing embedding.

        Args:
            embedding_id (int): The ID of the embedding.
            metadata (EmbeddingMetadataObj): The new metadata object.

        Returns:
            EmbeddingMetadataObj: The updated metadata object.
        """
        pass

    @abstractmethod
    def add_observation(
        self, embedding_id: int, observation: Tuple[float, int]
    ) -> None:
        """Atomically add an observation to an embedding's metadata.

        Args:
            embedding_id (int): The ID of the embedding.
            observation (Tuple[float, int]): A tuple (similarity_score, label).
        """
        pass

    @abstractmethod
    def remove_metadata(self, embedding_id: int) -> None:
        """Remove metadata for an embedding.

        Args:
            embedding_id (int): The ID of the embedding.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Clear all metadata from storage."""
        pass

    @abstractmethod
    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """Retrieve all metadata objects in storage.

        Returns:
            List[EmbeddingMetadataObj]: All metadata objects in storage.
        """
        pass
