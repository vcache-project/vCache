import random
from typing import Callable, Optional, Sequence, Tuple
from math import inf

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

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
        self.phi_inv: Optional[
            Callable[
                [
                    float,
                    np.ndarray,
                    np.ndarray,
                    float,
                    Callable[[float, np.ndarray, np.ndarray, float], float],
                    float,
                ],
                float,
            ]
        ] = self._normal_quantile

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

        if len(similarities) < 2 or len(labels) < 2:
            return Action.EXPLORE

        t_hat = self._estimate_parameters(similarities, labels, metadata)
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

    def _normal_quantile(
        self,
        t_hat: float,
        similarities: np.ndarray,
        labels: np.ndarray,
        quantile: float,
        loss_function: Callable[[float, np.ndarray, np.ndarray], float],
        gamma: float,
    ) -> float:
        alpha: float = 2 * (1.0 - quantile)
        _, upper = self._confidence_interval_fisher_method(
            t_hat=t_hat,
            sims=similarities,
            gamma=gamma,
            alpha=alpha,
        )
        return upper

    def _confidence_interval_delta_method(
        self,
        t_hat: float,
        sims: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        loss_function: Callable[[float, np.ndarray, np.ndarray], float],
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using delta method.
        """
        h = 1e-4
        # Variance estimation
        f0 = loss_function(t_hat, sims, labels)
        f1 = loss_function(t_hat + h, sims, labels)
        f2 = loss_function(t_hat - h, sims, labels)
        second_deriv = (f1 + f2 - 2 * f0) / (h * h)
        n = len(sims)
        var_t = 1.0 / (n * second_deriv + 1e-12)
        var_t = max(var_t, 1e-6)

        # Compute confidence interval
        z = norm.ppf(1 - alpha / 2)  # phi^-1(1 - alpha/2)
        delta = z * np.sqrt(var_t)
        return t_hat - delta, t_hat + delta

    def _confidence_interval_fisher_method(
        self,
        t_hat: float,
        sims: np.ndarray,
        gamma: float,
        alpha: float,
    ) -> Tuple[float, float]:
        # 1) compute p_i = L(s_i, t_hat)
        p = expit(gamma * (sims - t_hat))
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
        lower = t_hat - delta
        upper = t_hat + delta
        return lower, upper

    def _estimate_parameters(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        initial_t_guess = np.array([0.8])
        t_bounds = [(0.0, 1.0)]
        result = minimize(
            fun=lambda x: self._binary_cross_entropy_loss(
                t=x[0], sims=similarities, labels=labels, gamma=metadata.gamma
            ),
            x0=initial_t_guess,
            bounds=t_bounds,
            method="L-BFGS-B",
        )
        t_hat = float(result.x[0])

        return t_hat

    def _binary_cross_entropy_loss(
        self, t: float, sims: np.ndarray, labels: np.ndarray, gamma: float
    ) -> float:
        likelihood = self._likelihood(sims, t, gamma)
        eps = np.finfo(float).eps  # ≈2.2e−16
        likelihood = np.clip(likelihood, eps, 1.0 - eps)
        bce_loss = -np.mean(
            labels * np.log(likelihood) + (1 - labels) * np.log(1 - likelihood)
        )
        return bce_loss

    def _get_tau(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        s: float,
        t_hat: float,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        eps_array = np.array(self.epsilon_grid)
        quantiles = 1.0 - eps_array

        t_primes = np.array(
            [
                self.phi_inv(
                    t_hat=t_hat,
                    similarities=similarities,
                    labels=labels,
                    quantile=q,
                    loss_function=lambda t, sims, labs: self._binary_cross_entropy_loss(
                        t, sims, labs, metadata.gamma
                    ),
                    gamma=metadata.gamma,
                )
                for q in quantiles
            ]
        )

        likelihoods = self._likelihood(s, t_primes, metadata.gamma)
        alpha_lower_bounds = (1 - eps_array) * likelihoods
        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)

        return np.min(taus)

    def _likelihood(self, s: float, t_prime: float, gamma: float) -> float:
        z = gamma * (s - t_prime)
        return expit(z)
