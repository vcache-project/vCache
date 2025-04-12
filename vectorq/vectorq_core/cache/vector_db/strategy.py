import math
from typing import List, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_storage import EmbeddingMetadataStorageType
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig

class VectorDBStrategy(ABC):
    
    def __init__(
            self,
            vectorq_config: "VectorQConfig"
        ):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
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
