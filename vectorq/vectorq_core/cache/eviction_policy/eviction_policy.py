from abc import ABC, abstractmethod
from typing import List

class EvictionPolicy(ABC):
    
    @abstractmethod
    def rank(self, embeddings: List[str]) -> List[int]:
        '''
        embeddings: List[str] - The embeddings to rank
        returns: List[int] - The ranked embeddings
        '''
        pass
