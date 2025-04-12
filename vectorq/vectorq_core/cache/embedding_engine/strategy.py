from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
class EmbeddingEngineStrategy(ABC):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        '''
        text: str - The text to get the embedding for
        returns: List[float] - The embedding of the text
        '''
        pass
