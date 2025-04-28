import random
from typing import Callable, Optional, Sequence, Tuple, List
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
        self.epsilon_grid: np.ndarray = np.array([k / 100 for k in range(1, 50)])

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> Action:
        """
        Select the action to take based on the similarity score, observations, and accuracy target
        Args
            similarity_score: float - The similarity score between the query and the embedding
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            Action - Explore or Exploit
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
            similarities, similarity_score, t_hat, metadata
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
        Update the metadata with the new observation
        Args
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
            #result = model.fit_regularized(method='l2', alpha=0.1, disp=0, maxiter=100)
            
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
        s: float,
        t_hat: float,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        """
        Find the minimum tau value for the given similarity score
        Args
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            s: float - The similarity score between the query and the nearest neighbor
            t_hat: float - The estimated threshold
            metadata: EmbeddingMetadataObj - The metadata of the nearest neighbor
        Returns
            float - The minimum tau value
        """
        t_primes: List[float] = self._get_t_prime(t_hat=t_hat, similarities=similarities, gamma=metadata.gamma)
        likelihoods = self._likelihood(s=s, t=t_primes, gamma=metadata.gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods
        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        return np.min(taus)
    
    def _get_t_prime(
        self,
        t_hat: float,
        similarities: np.ndarray,
        gamma: float,
    ) -> List[float]:
        """
        Compute all possible t_prime values
        Args
            t_hat: float - The estimated threshold
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            gamma: float - The estimated gamma
        Returns
            List[float] - The t_prime values
        """
        t_primes: List[float] = np.array(
            [
                self._confidence_interval(
                    t_hat=t_hat,
                    similarities=similarities,
                    quantile=(1 - self.epsilon_grid[i]),
                    gamma=gamma,
                )
                for i in range(len(self.epsilon_grid))
            ]
        )
        return t_primes
    
    def _confidence_interval(
        self,
        t_hat: float,
        similarities: np.ndarray,
        quantile: float,
        gamma: float,
    ) -> float:
        """
        Compute the upper bound of the confidence interval for the given t_hat 
        with the Fisher method
        Args
            t_hat: float - The estimated threshold
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            gamma: float - The estimated gamma
        """
        alpha: float = 2 * (1.0 - quantile)
        # 1) compute p_i = L(s_i, t_hat)
        p = expit(gamma * (similarities - t_hat))
        # 2) observed Fisher information
        i = np.sum(gamma**2 * p * (1 - p))
        if i <= 0:
            # no information ⇒ infinite‐width interval
            return -inf, inf
        # 3) standard error
        se = np.sqrt(1.0 / i)
        # 4) normal quantile
        z = norm.ppf(1 - alpha / 2)
        delta = z * se
        _ = t_hat - delta
        upper = t_hat + delta
        return upper   

    def _likelihood(self, s: float, t: float, gamma: float) -> float:
        z = gamma * (s - t)
        return expit(z)
