from abc import ABC, abstractmethod
from typing import List


class EmbeddingEngine(ABC):
    """
    Abstract base class for embedding engines.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for the given text.

        Args:
            text: The text to get the embedding for.

        Returns:
            The embedding of the text as a list of floats.
        """
        raise NotImplementedError("Subclasses must implement this method")
