from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from vcache.vcache_core.cache.vector_db.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class SimilarityMetricType(Enum):
    """
    Enumeration of supported similarity metric types.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorDB(ABC):
    """
    Abstract base class for vector databases that store both embeddings and metadata.
    """

    def transform_similarity_score(
        self, similarity_score: float, metric_type: str
    ) -> float:
        """
        Transform similarity score to a normalized range.

        Args:
            similarity_score: The similarity score to transform.
            metric_type: The type of similarity metric.

        Returns:
            The transformed similarity score in the range of [0, 1].
        """
        match metric_type:
            case "cosine":
                return 1 - similarity_score
            case "euclidean":
                return 1 - similarity_score
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")

    @abstractmethod
    def add(self, embedding: List[float], metadata: EmbeddingMetadataObj) -> int:
        """
        Add an embedding and its metadata to the vector database.

        Args:
            embedding: The embedding to add to the vector db.
            metadata: The metadata object associated with the embedding.

        Returns:
            The id of the embedding.
        """
        pass

    @abstractmethod
    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding and its metadata from the vector database.

        Args:
            embedding_id: The id of the embedding to remove.

        Returns:
            The id of the embedding.
        """
        pass

    @abstractmethod
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """
        Get k-nearest neighbors for the given embedding.

        Args:
            embedding: The embedding to get the k-nearest neighbors for.
            k: The number of nearest neighbors to get.

        Returns:
            A list of tuples, each containing a similarity score and an embedding id.
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
    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in the database.

        Returns:
            A list of all the embedding metadata objects in the database.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the vector database to empty state.
        """
        pass

    @abstractmethod
    def _init_vector_store(self, embedding_dim: int):
        """
        Initialize the vector store with the given embedding dimension.

        Args:
            embedding_dim: The dimension of the embedding.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Check if the vector database is empty.

        Returns:
            True if the vector db is empty, False otherwise.
        """
        pass
