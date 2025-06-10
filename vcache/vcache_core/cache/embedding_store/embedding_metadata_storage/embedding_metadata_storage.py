from abc import ABC, abstractmethod
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class EmbeddingMetadataStorage(ABC):
    """
    Abstract base class for embedding metadata storage.
    """

    @abstractmethod
    def add_metadata(self, embedding_id: int, metadata: EmbeddingMetadataObj) -> int:
        """
        Add metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to add the metadata for.
            metadata: The metadata to add to the embedding.

        Returns:
            The id of the embedding.
        """
        pass

    @abstractmethod
    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to get the metadata for.

        Returns:
            The metadata of the embedding.
        """
        pass

    @abstractmethod
    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to update the metadata for.
            metadata: The metadata to update the embedding with.

        Returns:
            The updated metadata of the embedding.
        """
        pass

    @abstractmethod
    def remove_metadata(self, embedding_id: int) -> None:
        """
        Remove metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to remove the metadata for.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Flush all metadata from storage.
        """
        pass

    @abstractmethod
    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in storage.

        Returns:
            A list of all the embedding metadata objects in the storage.
        """
        pass
