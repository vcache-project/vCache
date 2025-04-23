from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy
from typing import Sequence, Optional, Callable, Tuple, List
import numpy as np
from scipy.stats import norm
import random
from scipy.optimize import minimize

class VectorQBayesianPolicy(VectorQPolicy):
    
    def __init__(self, delta: float):
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: Sequence[float] = [k / 100 for k in range(1, 50)]
        self.phi_inv: Optional[Callable[[float, np.ndarray, np.ndarray, float, Callable[[float, np.ndarray, np.ndarray, float], float]], float]] = self._normal_quantile
        
    def select_action(self, similarity_score: float, metadata: EmbeddingMetadataObj) -> Action:
        '''
        similarity_score: float - The similarity score between the query and the embedding
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        delta: float - Target correctness probability
        returns: Action - Explore or Exploit
        '''
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])
        
        if len(similarities) == 0 or len(labels) == 0:
            return Action.EXPLORE

        t_hat = self._estimate_parameters(similarities, labels, metadata)
        tau: float = self._get_tau(similarities, labels, similarity_score, t_hat, metadata)
        u: float = random.uniform(0, 1)
        
        if u <= tau:
            return Action.EXPLORE, t_hat, tau, u
        else:
            return Action.EXPLOIT, t_hat, tau, u
    
    def update_policy(self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj) -> None:
        '''
        similarity_score: float - The similarity score between the query and the embedding
        is_correct: bool - Whether the query was correct
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        '''
        if is_correct:
            metadata.observations.append((similarity_score, 1))
        else:
            metadata.observations.append((similarity_score, 0))

    def _normal_quantile(self, t_hat: float, similarities: np.ndarray, labels: np.ndarray, quantile: float, loss_function: Callable[[float, np.ndarray, np.ndarray], float]) -> float:
        alpha: float = 2 * (1.0 - quantile)
        _, upper = self._asymptotic_confidence_interval(t_hat, similarities, labels, alpha, loss_function)
        return upper

    def _asymptotic_confidence_interval(
        self,
        t_hat: float,
        sims: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        loss_function: Callable[[float, np.ndarray, np.ndarray], float]
    ) -> Tuple[float, float]:
        """
        Approximate a (1−alpha) confidence interval for t via
        the delta method (using numerical second derivative).
        """
        h = 1e-4
        f0 = loss_function(t_hat, sims, labels)
        f1 = loss_function(t_hat + h, sims, labels)
        f2 = loss_function(t_hat - h, sims, labels)
        second_deriv = (f1 + f2 - 2 * f0) / (h * h)
        n = len(sims)
        var_t = 1.0 / (n * second_deriv + 1e-12)
        z = norm.ppf(1 - alpha/2)
        delta = z * np.sqrt(var_t)
        return t_hat - delta, t_hat + delta
    
    def _estimate_parameters(self, similarities: np.ndarray, labels: np.ndarray, metadata: EmbeddingMetadataObj) -> float:
        initial_t_guess = np.array([0.8])
        t_bounds = [(0.0, 1.0)]
        result = minimize(
            fun=lambda x: self._binary_cross_entropy_loss(
                t=x[0], 
                sims=similarities, 
                labels=labels, 
                gamma=metadata.gamma
            ),
            x0=initial_t_guess,
            bounds=t_bounds,
            method='L-BFGS-B'
        )
        t_hat = float(result.x[0])
        
        return t_hat
    
    def _binary_cross_entropy_loss(self, t: float, sims: np.ndarray, labels: np.ndarray, gamma: float) -> float:
        likelihood = self._likelihood(sims, t, gamma)
        bce_loss = -np.mean(labels * np.log(likelihood) + (1 - labels) * np.log(1 - likelihood))
        return bce_loss
    
    def _get_tau(self, similarities: np.ndarray, labels: np.ndarray, s: float, t_hat: float, metadata: EmbeddingMetadataObj) -> float:
        taus: List[float] = []
        for eps in self.epsilon_grid:
            quantile: float = 1.0 - eps
            t_prime: float = self.phi_inv(
                t_hat, 
                similarities, 
                labels, 
                quantile, 
                lambda t, sims, labs: self._binary_cross_entropy_loss(t, sims, labs, metadata.gamma)
            )
            alpha_lower_bound: float = (1 - eps) * self._likelihood(s, t_prime, metadata.gamma)
            taus.append(self._approximate_tau(alpha_lower_bound))
        upper_lower_bound: float = min(taus)
        return upper_lower_bound

    def _likelihood(self, s: float, t_prime: float, gamma: float) -> float:
        z = gamma * (s - t_prime)
        return 1 / (1 + np.exp(-z))

    def _approximate_tau(self, alpha_lower_bound: float) -> float:
        return 1 - (1 - self.P_c) / (1 - alpha_lower_bound)
