from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy
from typing import Sequence, Optional, Callable, Tuple, List
import numpy as np
from scipy.stats import norm
import random
from scipy.optimize import minimize

class VectorQBayesianPolicy(VectorQPolicy):
    
    def __init__(self):
        self.epsilon_grid: Sequence[float] = [k / 100 for k in range(1, 50)]
        self.phi_inv: Optional[Callable[[float, np.ndarray, np.ndarray, float, Callable[[float, np.ndarray, np.ndarray, float], float]], float]] = self._normal_quantile
        
    def select_action(self, similarity_score: float, metadata: EmbeddingMetadataObj, delta: float) -> Action:
        '''
        similarity_score: float - The similarity score between the query and the embedding
        metadata: EmbeddingMetadataObj - The metadata of the embedding
        delta: float - Target correctness probability
        returns: Action - Explore or Exploit
        '''
        P_c: float = 1.0 - delta
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        t_hat, gamma = self._estimate_parameters(similarities, labels, metadata)
        metadata.gamma = gamma
        
        tau: float = self._get_tau(similarities, labels, similarity_score, t_hat, P_c, metadata)

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
        _, upper = self._asymptotic_ci(t_hat, similarities, labels, alpha, loss_function)
        return upper

    def _asymptotic_ci(
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
    
    def _estimate_parameters(self, similarities: np.ndarray, labels: np.ndarray, metadata: EmbeddingMetadataObj) -> Tuple[float, float]:
        """
        Estimate optimal t_hat parameter using gradient-based optimization.
        
        Returns:
            t_hat: Estimated threshold
        """
        # Initial guess for t
        x0 = np.array([0.5])
        
        # Bounds for parameter: t ∈ [0,1]
        bounds = [(0.0, 1.0)]
        
        result = minimize(
            fun=lambda x: self._binary_cross_entropy_loss(
                t=x[0], 
                sims=similarities, 
                labels=labels, 
                gamma=metadata.gamma, 
                apply_regularization=True
            ),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        t_hat = float(result.x[0])
        
        return t_hat, metadata.gamma
    
    def _joint_loss_function(self, t: float, gamma: float, sims: np.ndarray, labels: np.ndarray) -> float:
        """
        Binary cross-entropy loss with gamma as a parameter.
        """
        z: float = gamma * (sims - t)
        p: float = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)  # numeric safety
        reg_term: float = 0.01 * (gamma - 20.0)**2      # TODO: LGS validate
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)) + reg_term
    
    def _binary_cross_entropy_loss(self, t: float, sims: np.ndarray, labels: np.ndarray, gamma: float, apply_regularization: bool = False) -> float:
        """
        Binary cross-entropy loss with optional gamma parameter and regularization.
        
        Args:
            t: Threshold parameter
            sims: Similarity values
            labels: Binary labels
            gamma: Logistic slope
            apply_regularization: Whether to apply ridge regularization to gamma
            
        Returns:
            Loss value (BCE loss + optional regularization)
        """
        likelihood = self._likelihood(sims, t, gamma)
        bce_loss = -np.mean(labels * np.log(likelihood) + (1 - labels) * np.log(1 - likelihood))
        
        if apply_regularization and gamma is not None:
            #reg_term: float = 0.01 * gamma**2
            #return bce_loss + reg_term
            return bce_loss
        else:
            return bce_loss
    
    def _get_tau(self, similarities: np.ndarray, labels: np.ndarray, s: float, t_hat: float, P_c: float, metadata: EmbeddingMetadataObj) -> float:
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
            taus.append(self._approximate_tau(alpha_lower_bound, P_c))
        upper_lower_bound: float = min(taus)
        return upper_lower_bound

    def _likelihood(self, s: float, t_prime: float, gamma: float) -> float:
        z = gamma * (s - t_prime)
        return 1 / (1 + np.exp(-z))

    def _approximate_tau(self, alpha_lower_bound: float, P_c: float) -> float:
        return 1 - (1 - P_c) / (1 - alpha_lower_bound)
