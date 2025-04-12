from typing import Dict, Any, Optional, TYPE_CHECKING, List
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.strategy import EmbeddingMetadataStorageStrategy
if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
    
class InMemory(EmbeddingMetadataStorageStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config)
        self.metadata_storage: Dict[int, "EmbeddingMetadataObj"] = {}
        
    def add_metadata(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> int:
        self.metadata_storage[embedding_id] = metadata
        return embedding_id
    
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding metadata for embedding id {embedding_id} not found")
        else:
            return self.metadata_storage[embedding_id]
    
    def update(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding metadata for embedding id {embedding_id} not found")
        else:
            self.metadata_storage[embedding_id] = metadata
            return metadata
        
    def remove_metadata(self, embedding_id: int) -> None:
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding metadata for embedding id {embedding_id} not found")
        else:
            del self.metadata_storage[embedding_id]
    
    def flush(self) -> None:
        self.metadata_storage = {}
        
    def get_all_embedding_metadata_objects(self) -> List["EmbeddingMetadataObj"]:
        return list(self.metadata_storage.values())
