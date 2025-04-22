from abc import ABC, abstractmethod
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.vectorq_policy.action import Action

class VectorQPolicy(ABC):
    
    @abstractmethod
    def select_action(self, similarity_score: float, metadata: EmbeddingMetadataObj) -> Action:
        '''
        similarity_score: float - The similarity score between the query and the embedding
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        returns: Action - Explore or Exploit
        '''
        pass
    
    @abstractmethod
    def update_policy(self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj) -> None:
        '''
        similarity_score: float - The similarity score between the query and the embedding
        is_correct: bool - Whether the query was correct
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        '''
        pass
