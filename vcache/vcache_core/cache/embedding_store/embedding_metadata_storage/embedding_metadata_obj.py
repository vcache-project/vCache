from datetime import datetime
from typing import List, Tuple


class EmbeddingMetadataObj:
    """
    Metadata object for storing embedding-related information and statistics.
    """

    def __init__(
        self,
        embedding_id: int,
        response: str,
        last_accessed: datetime = None,
    ):
        """
        Initialize embedding metadata object.

        Args:
            embedding_id: Unique identifier for the embedding.
            response: The response associated with the embedding.
            prior: Prior distribution for Bayesian inference.
            posterior: Posterior distribution for Bayesian inference.
            region_reject: List of rejection regions for heuristic policy.
            last_accessed: Timestamp of last access to this embedding.
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
        self.gamma: float = None
        self.t_hat: float = None

        #### Metadata for the eviction policy ################################
        self.last_accessed: datetime = last_accessed
        self.usage_count: int = 0

    def __eq__(self, other):
        """
        Check equality with another EmbeddingMetadataObj.

        Args:
            other: The other object to compare with.

        Returns:
            True if objects are equal, False otherwise.
        """
        if not isinstance(other, EmbeddingMetadataObj):
            return False
        return (
            self.embedding_id == other.embedding_id
            and self.response == other.response
            and self.last_accessed == other.last_accessed
        )

    def __repr__(self):
        """
        Return string representation of the embedding metadata object.

        Returns:
            String representation of the object.
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
