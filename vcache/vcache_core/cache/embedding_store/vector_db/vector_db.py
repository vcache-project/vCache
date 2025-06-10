from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class SimilarityMetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorDB(ABC):
    def transform_similarity_score(
        self, similarity_score: float, metric_type: str
    ) -> float:
        """
        Transforms the similarity score to the range of [0, 1] based on the similarity metric type.

        Args:
            similarity_score: float - The similarity score to transform
            metric_type: SimilarityMetricType - The type of similarity metric

        Returns:
            float - The transformed similarity score in the range of [0, 1]
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
        Thread-safe addition of embedding to the vector database.

        Args:
            embedding: List[float] - The embedding vector to add

        Returns:
            int - The unique ID assigned to the embedding
        """
        pass

    @abstractmethod
    def remove(self, embedding_id: int) -> int:
        """
        Thread-safe removal of embedding from the vector database.

        Args:
            embedding_id: int - The ID of the embedding to remove

        Returns:
            int - The ID of the removed embedding
        """
        pass

    @abstractmethod
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """
        Thread-safe k-nearest neighbors search.

        Args:
            embedding: List[float] - The query embedding
            k: int - Number of nearest neighbors to return

        Returns:
            List[tuple[float, int]] - List of (similarity_score, embedding_id) tuples
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Thread-safe reset of the vector database.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _init_vector_store(self, embedding_dim: int):
        """
        Initializes the vector store. Should be called within a lock context.

        Args:
            embedding_dim: int - The dimension of the embedding vectors
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Thread-safe check if the vector database is empty.

        Returns:
            bool - True if the database is empty, False otherwise
        """
        pass
