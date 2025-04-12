from typing import Dict, Any, Optional, TYPE_CHECKING, List
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.strategy import EmbeddingMetadataStorageStrategy

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
class LangChain(EmbeddingMetadataStorageStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config)
        
    def add_metadata(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> int:
        # TODO
        return embedding_id
    
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        # TODO
        return None
    
    def update(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        # TODO
        return None
    
    def remove_metadata(self, embedding_id: int) -> None:
        # TODO
        pass
    
    def flush(self) -> None:
        # TODO
        pass
    
    def get_all_embedding_metadata_objects(self) -> List["EmbeddingMetadataObj"]:
        # TODO
        return []
    