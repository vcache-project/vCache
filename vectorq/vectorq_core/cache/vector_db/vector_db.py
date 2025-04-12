from enum import Enum
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.cache.vector_db.strategies.faiss import FAISS
from vectorq.vectorq_core.cache.vector_db.strategies.chroma import Chroma
from vectorq.vectorq_core.cache.vector_db.strategies.hnsw_lib import HNSWLib
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage import EmbeddingMetadataStorage

class VectorDBType(Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    HNSW = "hnsw"
    
class SimilarityMetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class VectorDB():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.vector_db_strategy = None
        match self.vectorq_config._vector_db_type:
            case VectorDBType.FAISS:
                self.vector_db_strategy = FAISS(self.vectorq_config)
            case VectorDBType.CHROMA:
                self.vector_db_strategy = Chroma(self.vectorq_config)
            case VectorDBType.HNSW:
                self.vector_db_strategy = HNSWLib(self.vectorq_config)
            case _:
                raise ValueError(f"Invalid vector db type")
            
        self.embedding_metadata_storage = EmbeddingMetadataStorage(self.vectorq_config)
    
    def add_embedding(self, embedding: List[float], response: str) -> int:
        embedding_id: int = self.vector_db_strategy.add(embedding)
        metadata: "EmbeddingMetadataObj" = EmbeddingMetadataObj(
            embedding_id=embedding_id,
            response=response,
        )
        self.embedding_metadata_storage.add_metadata(embedding_id=embedding_id, metadata=metadata)
        return embedding_id
    
    def remove(self, embedding_id: int) -> int:
        self.embedding_metadata_storage.remove_metadata(embedding_id)
        return self.vector_db_strategy.remove(embedding_id)
    
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        return self.vector_db_strategy.get_knn(embedding, k)
    
    def reset(self) -> None:
        self.embedding_metadata_storage.flush()
        return self.vector_db_strategy.reset()
    
    def calculate_storage_consumption(self) -> int:
        # TODO: Add metadata logic
        return -1
    
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        return self.embedding_metadata_storage.get_metadata(embedding_id)
    
    def update(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        return self.embedding_metadata_storage.update(embedding_id, metadata)

    def is_empty(self) -> bool:
        return self.vector_db_strategy.embedding_count == 0