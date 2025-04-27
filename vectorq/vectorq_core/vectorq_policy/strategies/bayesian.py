import random
from typing import Callable, Optional, Sequence, Tuple
from math import inf

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy


class VectorQBayesianPolicy(VectorQPolicy):
    def __init__(self, delta: float):
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: Sequence[float] = [k / 100 for k in range(1, 50)]

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> Action:
        """
        similarity_score: float - The similarity score between the query and the embedding
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        delta: float - Target correctness probability
        returns: Action - Explore or Exploit
        """
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 4 or len(labels) < 4:
            return Action.EXPLORE

        t_hat, gamma = self._estimate_parameters(similarities=similarities, labels=labels)
        if t_hat == -1:
            return Action.EXPLORE
        metadata.gamma = gamma
        metadata.t_hat = t_hat
        tau: float = self._get_tau(
            similarities, labels, similarity_score, t_hat, metadata
        )
        u: float = random.uniform(0, 1)
        if u <= tau:
            return Action.EXPLORE
        else:
            return Action.EXPLOIT

    def update_policy(
        self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj
    ) -> None:
        """
        similarity_score: float - The similarity score between the query and the embedding
        is_correct: bool - Whether the query was correct
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        """
        if is_correct:
            metadata.observations.append((similarity_score, 1))
        else:
            metadata.observations.append((similarity_score, 0))
            

    def _estimate_parameters(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Optimize parameters with logistic regression
        Args
            similarities: np.ndarray - The similarities of the embeddings
            labels: np.ndarray - The labels of the embeddings
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            t_hat: float - The estimated threshold
            gamma: float - The estimated gamma
        """
        
        X = sm.add_constant(similarities)
        
        try:
            model = sm.Logit(labels, X)
            result = model.fit(disp=0)
            
            intercept = result.params[0]
            gamma     = result.params[1]
            
            if abs(gamma) < 1e-5:
                gamma = 1.0 if gamma >= 0 else -1.0
                
            t_hat = -intercept / gamma
            
            t_hat = max(0.0, min(1.0, t_hat))
            gamma = max(10, gamma)
            
            return t_hat, gamma
            
        except Exception as e:
            print(f"Logistic regression failed: {e}")
            return -1.0, -1.0

    def _get_tau(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        s: float,
        t_hat: float,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        eps_array = np.array(self.epsilon_grid)

        likelihoods = self._likelihood(s, t_hat, metadata.gamma)
        alpha_lower_bounds = (1 - eps_array) * likelihoods
        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)

        return np.min(taus)

    def _likelihood(self, s: float, t_prime: float, gamma: float) -> float:
        z = gamma * (s - t_prime)
        return expit(z)
