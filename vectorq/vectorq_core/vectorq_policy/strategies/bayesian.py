import logging
import random
import time
from typing import List, Tuple

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy


class VectorQBayesianPolicy(VectorQPolicy):
    def __init__(self, delta: float):
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: np.ndarray = np.array([k / 100 for k in range(1, 50, 5)])
        self.logistic_regression: LogisticRegression = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )
        self.logistic_regression_regularized: LogisticRegression = LogisticRegression(
            penalty=None,  #'l1',
            solver="lbfgs",
            # C=0.01,
            tol=1e-6,
            max_iter=1000,
            fit_intercept=False,
        )

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

        if len(similarities) < 6 or len(labels) < 6:
            return Action.EXPLORE

        start_time = time.time()
        t_hat, gamma = self._estimate_parameters(
            similarities=similarities, labels=labels
        )
        logging.info(f"Duration para est.: {(time.time() - start_time):.4f} sec")
        if t_hat == -1:
            return Action.EXPLORE
        metadata.gamma = gamma
        metadata.t_hat = t_hat

        start_time = time.time()
        tau: float = self._get_tau(
            similarities, labels, similarity_score, t_hat, metadata
        )
        logging.info(f"t_hat: {t_hat:.4f}, gamma: {gamma:.4f}, tau: {tau:.4f}")
        logging.info(f"Duration tau est.: {(time.time() - start_time):.4f} sec\n")

        u: float = random.uniform(0, 1)
        if u <= tau:
            return Action.EXPLORE
        else:
            return Action.EXPLOIT

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
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

        similarities = sm.add_constant(similarities)

        try:
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]
            t_hat = -intercept / (gamma + 1e-6)
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
        """
        Find the minimum tau value for the given similarity score
        Args
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            labels: np.ndarray - The labels in respect to the similarities
            s: float - The similarity score between the query and the nearest neighbor
            t_hat: float - The estimated threshold
            metadata: EmbeddingMetadataObj - The metadata of the nearest neighbor
        Returns
            float - The minimum tau value
        """
        # Currently: 0.10330510139465332 seconds
        # start_time = time.time()
        t_primes: List[float] = self._get_t_prime(
            t_hat=t_hat, similarities=similarities, labels=labels
        )
        # end_time = time.time()
        # print(f"Time taken to get t_primes: {end_time - start_time} seconds")
        likelihoods = self._likelihood(s=s, t=t_primes, gamma=metadata.gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods
        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        return np.min(taus)

    def _get_t_prime(
        self, t_hat: float, similarities: np.ndarray, labels: np.ndarray
    ) -> List[float]:
        """
        Compute all possible t_prime values
        Args
            t_hat: float - The estimated threshold
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            labels: np.ndarray - The labels in respect to the similarities
            gamma: float - The estimated gamma
        Returns
            List[float] - The t_prime values
        """
        similarities = similarities.reshape(-1, 1)
        similarities = sm.add_constant(similarities)
        var_t = self.__bootstrap_variance(similarities, labels)
        # var_t = 0.01
        t_primes: List[float] = np.array(
            [
                self._confidence_interval(
                    t_hat=t_hat, var_t=var_t, quantile=(1 - self.epsilon_grid[i])
                )
                for i in range(len(self.epsilon_grid))
            ]
        )
        logging.info(f"var_t: {var_t}, t_primes: {t_primes}")
        return t_primes

    def __bootstrap_variance(
        self, similarities: np.ndarray, labels: np.ndarray, n_bootstraps: int = 128
    ):
        """
        Compute the variance of t with bootstrapping
        Args
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            labels: np.ndarray - The labels of the embeddings
            n_bootstraps: int - The number of bootstraps (Default: 256)
        Returns
            float - The variance of t
        """
        t_boots = np.array(
            [self.__one_bootstrap(similarities, labels) for _ in range(n_bootstraps)]
        )
        t_boots = t_boots[(t_boots > 0) & (t_boots < 1)]
        if t_boots.size == 0:
            return 0.01
        var_t = max(0.0, min(1.0, np.nanvar(t_boots)))
        return var_t

    def __one_bootstrap(self, similarities: np.ndarray, labels: np.ndarray):
        """
        Compute the t value with one bootstrap
        Args
            similarities: np.ndarray - The similarities observed for the nearest neighbor
            labels: np.ndarray - The labels of the embeddings
        Returns
            float - The t value
        """
        similaritiesb, labelsb = resample(similarities, labels, replace=True)
        if labelsb.min() == labelsb.max():
            return np.nan
        self.logistic_regression_regularized.fit(similaritiesb, labelsb)
        intercept, gamma = self.logistic_regression_regularized.coef_[0]
        t = -intercept / (gamma + 1e-6)
        return t

    def _confidence_interval(
        self, t_hat: float, var_t: float, quantile: float
    ) -> float:
        """
        Return the (upper) quantile-threshold t' such that
          P_est( t > t' ) <= 1 - quantile
        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t
            quantile: float - The quantile
        Returns
            float - The t_prime value
        """
        z = norm.ppf(quantile)
        t_prime = t_hat + z * np.sqrt(var_t)
        return float(np.clip(t_prime, 0.0, 1.0))

    def _likelihood(self, s: float, t: float, gamma: float) -> float:
        z = gamma * (s - t)
        return expit(z)
