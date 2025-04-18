from abc import ABC, abstractmethod
from typing import List

class EmbeddingEngine(ABC):
    """
    Abstract base class for embedding engines
    """

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        '''
        text: str - The text to get the embedding for
        returns: List[float] - The embedding of the text
        '''
        raise NotImplementedError("Subclasses must implement this method")
