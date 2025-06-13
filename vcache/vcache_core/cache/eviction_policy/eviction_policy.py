from abc import ABC, abstractmethod
from typing import List


class EvictionPolicy(ABC):
    """
    Abstract base class for cache eviction policies.
    """

    @abstractmethod
    def rank(self, embeddings: List[str]) -> List[int]:
        """
        Rank embeddings for eviction priority.

        Args:
            embeddings: The embeddings to rank.

        Returns:
            The ranked embeddings by eviction priority.
        """
        pass
