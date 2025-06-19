from datetime import datetime, timezone
from typing import List, Optional, Tuple


class EmbeddingMetadataObj:
    """
    Metadata object for storing embedding-related information and statistics.
    """

    def __init__(
        self,
        embedding_id: int,
        response: str,
        last_accessed: Optional[datetime] = None,
    ):
        """Initializes the embedding metadata object.

        Args:
            embedding_id (int): The unique identifier for the embedding.
            response (str): The response associated with the embedding.
            last_accessed (Optional[datetime]): The timestamp of the last access
                to this embedding. If not provided, the current time in UTC is
                used.
        """

        #### Core metadata ###################################################
        self.embedding_id: int = embedding_id
        self.response: str = response

        #### vCache Bayesian Policy ##########################################
        self.observations: List[Tuple[float, int]] = []  # (similarity, label)
        self.observations.append((0.0, 0))
        self.observations.append((1.0, 1))
        self.gamma: float = None
        self.t_hat: float = None
        self.t_prime: float = None
        self.var_t: float = None

        #### Metadata for the eviction policy ################################
        self.last_accessed: Optional[datetime] = (
            last_accessed if last_accessed is not None else datetime.now(timezone.utc)
        )
        self.created_at: datetime = datetime.now(timezone.utc)
        self.usage_count: int = 0

    def __eq__(self, other: object) -> bool:
        """Checks for equality with another EmbeddingMetadataObj.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, EmbeddingMetadataObj):
            return False
        return (
            self.embedding_id == other.embedding_id
            and self.response == other.response
            and self.last_accessed == other.last_accessed
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns:
            str: A string representation of the EmbeddingMetadataObj.
        """
        return f"""
        EmbeddingMetadataObj(
            embedding_id={self.embedding_id},
            response={self.response},
            last_accessed={self.last_accessed},
            len(observations)={len(self.observations)},
            observations={self.observations},
            gamma={self.gamma},
        )
        """
