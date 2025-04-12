from typing import Dict, Any, List, Optional, TYPE_CHECKING
from vectorq.vectorq_core.cache.vector_db.strategy import VectorDBStrategy

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.vectorq_core.cache.vector_db.vector_db import SimilarityMetricType
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_storage import EmbeddingMetadataStorageType
    from vectorq.config import VectorQConfig
    
class Custom(VectorDBStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config)
        self.embedding_count = 0
    
    def add(self, embedding: List[float], response: str) -> int:
        # TODO
        return 0
    
    def remove(self, embedding_id: int) -> int:
        # TODO
        return 0
    
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        # TODO
        return []
    
    def reset(self) -> None:
        # TODO
        pass
    
    def _init_vector_store(self, embedding_dim: int):
        # TODO
        pass
