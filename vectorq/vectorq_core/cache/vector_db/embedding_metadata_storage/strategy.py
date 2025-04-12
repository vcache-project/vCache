from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
    
class EmbeddingMetadataStorageStrategy(ABC):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
    @abstractmethod
    def add_metadata(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> int:
        '''
        embedding_id: int - The id of the embedding to add the metadata for
        metadata: EmbeddingMetadataObj - The metadata to add to the embedding
        returns: int - The id of the embedding
        '''
        pass

    @abstractmethod
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        '''
        embedding_id: int - The id of the embedding to get the metadata for
        returns: EmbeddingMetadataObj - The metadata of the embedding
        '''
        pass
    
    @abstractmethod
    def update(self, embedding_id: int, metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        '''
        embedding_id: int - The id of the embedding to update the metadata for
        metadata: EmbeddingMetadataObj - The metadata to update the embedding with
        returns: EmbeddingMetadataObj - The updated metadata of the embedding
        '''
        pass
    
    @abstractmethod
    def remove_metadata(self, embedding_id: int) -> None:
        '''
        embedding_id: int - The id of the embedding to remove the metadata for
        '''
        pass
    
    @abstractmethod
    def flush(self) -> None:
        '''
        Flushes the metadata storage
        '''
        pass

    @abstractmethod
    def get_all_embedding_metadata_objects(self) -> List["EmbeddingMetadataObj"]:
        '''
        returns: List["EmbeddingMetadataObj"] - A list of all the embedding metadata objects in the storage
        '''
        pass