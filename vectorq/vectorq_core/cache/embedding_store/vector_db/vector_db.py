from enum import Enum
from typing import List
from abc import ABC, abstractmethod

class SimilarityMetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class VectorDB(ABC):
    
    def transform_similarity_score(self, similarity_score: float, metric_type: str) -> float:
        '''
        similarity_score: float - The similarity score to transform
        metric_type: SimilarityMetricType - The type of similarity metric
        returns: float - The transformed similarity score in the range of [0, 1]
        '''
        match metric_type:
            case 'cosine':
                return 1 - similarity_score
            case 'euclidean':
                return 1 - similarity_score
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
    
    @abstractmethod
    def add(self, embedding: List[float]) -> int:
        '''
        embedding: List[float] - The embedding to add to the vector db
        returns: int - The id of the embedding
        '''
        pass
    
    @abstractmethod
    def remove(self, embedding_id: int) -> int:
        '''
        embedding_id: int - The id of the embedding to remove
        returns: int - The id of the embedding
        '''
        pass
    
    @abstractmethod
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        '''
        embedding: List[float] - The embedding to get the k-nearest neighbors for
        k: int - The number of nearest neighbors to get
        returns: List[tuple[float, int]] - A list of tuples, each containing a similarity score and an embedding id
        '''
        pass
    
    @abstractmethod
    def reset(self) -> None:
        '''
        Resets the vector db
        '''
        pass
    
    @abstractmethod
    def _init_vector_store(self, embedding_dim: int):
        '''
        embedding_dim: int - The dimension of the embedding
        '''
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        '''
        Returns: bool - Whether the vector db is empty
        '''
        pass
