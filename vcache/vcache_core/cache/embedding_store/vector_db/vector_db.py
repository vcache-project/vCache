from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class SimilarityMetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorDB(ABC):
    """Abstract base class for vector database implementations."""

    def transform_similarity_score(
        self, similarity_score: float, metric_type: str
    ) -> float:
        """Transform a distance-based score into a normalized similarity score.

        This function converts raw distance scores from vector databases (like
        Euclidean or cosine distance) into a unified similarity score from 0 to 1,
        where 1 means most similar.

        Args:
            similarity_score (float): The raw distance score from the vector database.
            metric_type (str): The similarity metric used (e.g., 'cosine', 'euclidean').

        Returns:
            float: The transformed similarity score, normalized to the range [0, 1].
        """
        match metric_type:
            case "cosine":
                return 1 - similarity_score
            case "euclidean":
                return 1 - similarity_score
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")

    @abstractmethod
    def add(self, embedding: List[float]) -> int:
        """Add an embedding to the vector database.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the embedding.
        """
        pass

    @abstractmethod
    def remove(self, embedding_id: int) -> int:
        """Remove an embedding from the vector database.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        pass

    @abstractmethod
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Find the k-nearest neighbors for a given embedding.

        Args:
            embedding (List[float]): The query embedding.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: A list of (similarity_score, embedding_id) tuples.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all embeddings from the vector database."""
        pass

    @abstractmethod
    def _init_vector_store(self, embedding_dim: int):
        """Initialize the underlying vector store.

        This method should be called within a lock and is responsible for setting up
        the database with the correct dimensions and parameters.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the vector database contains no embeddings.

        Returns:
            bool: True if the database is empty, False otherwise.
        """
        pass
