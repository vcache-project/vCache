from typing import Dict, Any, Optional, TYPE_CHECKING, List
from enum import Enum
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.strategy import EmbeddingMetadataStorageStrategy
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.strategies.lang_chain import LangChain
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.strategies.in_memory import InMemory
if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
    
class EmbeddingMetadataStorageType(Enum):
    LANGCHAIN = "langchain"
    SQL = "sql"
    FILE = "file"
    IN_MEMORY = "in_memory"

class EmbeddingMetadataStorage():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.strategy: EmbeddingMetadataStorageStrategy = None
        match self.vectorq_config._embedding_metadata_storage_type:
            case EmbeddingMetadataStorageType.LANGCHAIN:
                self.strategy: EmbeddingMetadataStorageStrategy = LangChain(self.vectorq_config)
            case EmbeddingMetadataStorageType.IN_MEMORY:
                self.strategy: EmbeddingMetadataStorageStrategy = InMemory(self.vectorq_config)
            case _:
                raise ValueError(f"Invalid embedding metadata storage type: {self.vectorq_config._embedding_metadata_storage_type}")
            
    def add_metadata(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> int:
        return self.strategy.add_metadata(embedding_id, metadata)
    
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        return self.strategy.get_metadata(embedding_id)
    
    def update(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        return self.strategy.update(embedding_id, metadata)
    
    def remove_metadata(self, embedding_id: int) -> None:
        return self.strategy.remove_metadata(embedding_id)
    
    def flush(self) -> None:
        self.strategy.flush()
        
    def get_all_embedding_metadata_objects(self) -> List["EmbeddingMetadataObj"]:
        return self.strategy.get_all_embedding_metadata_objects()
    