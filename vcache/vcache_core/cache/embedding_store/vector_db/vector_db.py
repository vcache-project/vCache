from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class SimilarityMetricType(Enum):
    """
    Enumeration of supported similarity metric types.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorDB(ABC):
    """
    Abstract base class for vector databases.
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
    def add(self, embedding: List[float]) -> int:
        """
        Add an embedding to the vector database.

        Args:
            embedding: The embedding to add to the vector db.

        Returns:
            The id of the embedding.
        """
        pass

    @abstractmethod
    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding from the vector database.

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

    @abstractmethod
    def size(self) -> int:
        """
        Get the number of embeddings in the vector database.

        Returns:
            The number of embeddings in the vector database.
        """
        pass
