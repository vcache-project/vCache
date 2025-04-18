from typing import Dict, Any, List, Optional

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import EmbeddingMetadataStorage

class InMemoryEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    """
    A simple in-memory implementation of EmbeddingMetadataStorage
    """
    
    def __init__(self):
        self.storage = {}  # embedding_id -> {"response": str, "metadata": Dict}
    
    def add_metadata(self, embedding_id: int, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add metadata for an embedding
        
        Args:
            embedding_id: The ID of the embedding
            response: The response associated with the embedding
            metadata: Additional metadata
        """
        self.storage[embedding_id] = {
            "response": response,
            "metadata": metadata or {}
        }
    
    def get_metadata(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an embedding
        
        Args:
            embedding_id: The ID of the embedding
            
        Returns:
            The metadata for the embedding, or None if not found
        """
        if embedding_id not in self.storage:
            return None
        
        data = self.storage[embedding_id]
        return {
            "response": data["response"],
            "metadata": data["metadata"].copy()
        }
    
    def remove_metadata(self, embedding_id: int) -> bool:
        """
        Remove metadata for an embedding
        
        Args:
            embedding_id: The ID of the embedding
            
        Returns:
            True if successful, False otherwise
        """
        if embedding_id in self.storage:
            del self.storage[embedding_id]
            return True
        return False
    
    def update(self, embedding_id: int, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update metadata for an embedding
        
        Args:
            embedding_id: The ID of the embedding
            response: The new response, or None to keep the existing response
            metadata: The new metadata, or None to keep the existing metadata
            
        Returns:
            True if successful, False otherwise
        """
        if embedding_id not in self.storage:
            return False
        
        if response is not None:
            self.storage[embedding_id]["response"] = response
            
        if metadata is not None:
            self.storage[embedding_id]["metadata"] = metadata
            
        return True
    
    def flush(self) -> None:
        """
        Reset the storage, removing all metadata
        """
        self.storage.clear()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects
        """
        return list(self.storage.values())
