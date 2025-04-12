from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.vector_db.vector_db import VectorDB
from vectorq.vectorq_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

class Cache:
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        self.vector_db = VectorDB(self.vectorq_config)
        self.eviction_policy = EvictionPolicy(self.vectorq_config)
        self.embedding_engine = EmbeddingEngine(self.vectorq_config)
    
    def add(self, prompt: str, response: str) -> int:
        '''
        prompt: str - The prompt to add to the cache
        response: str - The response to add to the cache
        returns: int - The id of the embedding
        '''
        embedding = self.embedding_engine.get_embedding(prompt)
        self.vector_db.add_embedding(embedding, response)
    
    def add_embedding(self, embedding: List[float], response: str) -> int:
        '''
        embedding: List[float] - The embedding to add to the cache
        response: str - The response to add to the cache
        returns: int - The id of the embedding
        '''
        self.vector_db.add_embedding(embedding, response)
    
    def remove(self, embedding_id: int) -> int:
        '''
        embedding_id: int - The id of the embedding to remove
        returns: int - The id of the embedding
        '''
        self.vector_db.remove(embedding_id)
    
    def update(self, embedding_id: int, embedding_metadata: "EmbeddingMetadataObj") -> "EmbeddingMetadataObj":
        '''
        embedding_id: int - The id of the embedding to update
        embedding_metadata: EmbeddingMetadataObj - The metadata to update the embedding with
        returns: EmbeddingMetadataObj - The updated metadata of the embedding
        '''
        self.vector_db.update(embedding_id, embedding_metadata)
    
    def get_knn(self, prompt: str, k: int, embedding: List[float] = []) -> List[tuple[float, int]]:
        '''
        prompt: str - The prompt to get the k-nearest neighbors for
        k: int - The number of nearest neighbors to get
        returns: List[tuple[float, int]] - A list of tuples, each containing a similarity score and an embedding id
        '''
        if (embedding == []):
            embedding: List[float] = self.embedding_engine.get_embedding(prompt)
        return self.vector_db.get_knn(embedding, k)
    
    def flush(self) -> None:
        '''
        Flushes the cache
        '''
        self.vector_db.flush()
    
    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        '''
        embedding_id: int - The id of the embedding to get the metadata for
        returns: EmbeddingMetadataObj - The metadata of the embedding
        '''
        return self.vector_db.get_metadata(embedding_id)
    
    def get_current_capacity(self) -> int:
        '''
        returns: int - The current capacity of the cache
        '''
        # TODO
        return None
    
    def is_empty(self) -> bool:
        '''
        returns: bool - Whether the cache is empty
        '''
        return self.vector_db.is_empty()
    
    def get_all_embedding_metadata_objects(self) -> List["EmbeddingMetadataObj"]:
        '''
        returns: List["EmbeddingMetadataObj"] - A list of all the embedding metadata objects in the cache
        '''
        return self.vector_db.embedding_metadata_storage.get_all_embedding_metadata_objects()
