from typing import List, Tuple
import numpy as np
from datetime import datetime

class EmbeddingMetadataObj:
    
    def __init__(
            self,
            embedding_id: int,
            response: str,
            prior: np.ndarray = None,
            posterior: np.ndarray = None,
            region_reject: List[str] = None,
            last_accessed: datetime = None,
    ):
        self.embedding_id: int = embedding_id
        self.response: str = response
        self.last_accessed: datetime = last_accessed
        
        # VectorQ Bayesian Policy ########################
        self.observations: List[Tuple[float, int]] = []     # (similarity, label)
        self.gamma: float = 20.0                            # Initial guess 
        ##################################################
        
        # VectorQ Heuristic Policy #######################
        self.prior: np.ndarray = prior
        self.posterior: np.ndarray = posterior
        self.region_reject: List[float] = region_reject
        self.correct_similarities: List[float] = []
        self.incorrect_similarities: List[float] = []
        self.posteriors: List[float] = []
        ################################################## 
        
    def __eq__(self, other):
        if not isinstance(other, EmbeddingMetadataObj):
            return False
        return (
            self.embedding_id == other.embedding_id and
            self.response == other.response and
            np.array_equal(self.prior, other.prior) and
            np.array_equal(self.posterior, other.posterior) and
            self.region_reject == other.region_reject and
            self.last_accessed == other.last_accessed
        )
        
    def __repr__(self):
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
        
    def add_correct_similarity(self, similarity: float):
        self.correct_similarities.append(similarity)
        
    def add_incorrect_similarity(self, similarity: float):
        self.incorrect_similarities.append(similarity)
        