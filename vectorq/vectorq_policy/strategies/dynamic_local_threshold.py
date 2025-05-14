import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from vectorq.config import VectorQConfig
from vectorq.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore
from vectorq.vectorq_core.similarity_evaluator import SimilarityEvaluator
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


class DynamicLocalThresholdPolicy(VectorQPolicy):
    def __init__(self, delta: float = 0.01):
        """
        This policy uses the VectorQ algorithm to compute the optimal threshold for each
        embedding in the cache.
        Each threshold is used to determine if a response is a cache hit.

        Args
            delta: float - The delta value to use
        """
        self.bayesian = _Bayesian(delta=delta)
        self.similarity_evaluator: SimilarityEvaluator = None
        self.inference_engine: InferenceEngine = None
        self.cache: Cache = None

    @override
    def setup(self, config: VectorQConfig):
        self.similarity_evaluator = config.similarity_evaluator
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )

    @override
    def process_request(
        self, prompt: str, system_prompt: Optional[str], set_id: Optional[str]
    ) -> tuple[bool, str, str, str, str]:
        """
        Args
            prompt: str - The prompt to check for cache hit
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        Returns
            tuple[bool, str, str, str, str] - [is_cache_hit, actual_response, nn_response, cache_response_set_id, nn_response_set_id]
        """
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")
        knn = self.cache.get_knn(prompt=prompt, k=1)
        if not knn:
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response, set_id=set_id)
            return False, response, "", set_id, -1

        similarity_score, embedding_id = knn[0]
        metadata = self.cache.get_metadata(embedding_id=embedding_id)
        action = self.bayesian.select_action(
            similarity_score=similarity_score, metadata=metadata
        )

        match action:
            case _Action.EXPLOIT:
                return True, metadata.response, metadata.response, metadata.set_id, metadata.set_id
            case _Action.EXPLORE:
                response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )
                should_have_exploited = self.similarity_evaluator.answers_similar(
                    a=set_id, b=metadata.set_id
                )
                self.bayesian.update_metadata(
                    similarity_score=similarity_score,
                    is_correct=should_have_exploited,
                    metadata=metadata,
                )
                if not should_have_exploited:
                    self.cache.add(prompt=prompt, response=response, set_id=set_id)
                self.cache.update_metadata(
                    embedding_id=embedding_id, embedding_metadata=metadata
                )
                return False, response, metadata.response, set_id, metadata.set_id


class _Action(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Bayesian:
    def __init__(self, delta: float):
        self.delta: float = delta
        self.P_c: float = 1.0 - self.delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.logistic_regression: LogisticRegression = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
        )

        self.variance_map: Dict[int, List[float]] = {
            6: 0.035445,
            7: 0.028285,
            8: 0.026436,
            9: 0.021349,
            10: 0.019371,
            11: 0.012615,
            12: 0.011433,
            13: 0.010228,
            14: 0.009963,
            15: 0.009253,
            16: 0.011674,
            17: 0.013015,
            18: 0.010897,
            19: 0.011841,
            20: 0.013081,
            21: 0.010585,
            22: 0.014255,
            23: 0.012058,
            24: 0.013002,
            25: 0.011715,
            26: 0.00839,
            27: 0.008839,
            28: 0.010628,
            29: 0.009899,
            30: 0.008033,
            31: 0.00457,
            32: 0.007335,
            33: 0.008932,
            34: 0.00729,
            35: 0.007445,
            36: 0.00761,
            37: 0.011423,
            38: 0.011233,
            39: 0.006783,
            40: 0.005233,
            41: 0.00872,
            42: 0.010005,
            43: 0.01199,
            44: 0.00977,
            45: 0.01891,
            46: 0.01513,
            47: 0.02109,
            48: 0.01531,
        }

    def update_metadata(
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
            metadata.observations.append((round(similarity_score, 3), 1))
        else:
            metadata.observations.append((round(similarity_score, 3), 0))

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> _Action:
        """
        Select the action to take based on the similarity score, observations, and accuracy target
        Args
            similarity_score: float - The similarity score between the query and the embedding
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            Action - Explore or Exploit
        """
        similarity_score = round(similarity_score, 3)
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 5 or len(labels) < 5:
            return _Action.EXPLORE

        t_hat, gamma, var_t = self._estimate_parameters(
            similarities=similarities, labels=labels
        )

        if t_hat == -1:
            return _Action.EXPLORE
        metadata.gamma = gamma
        metadata.t_hat = t_hat
        metadata.var_t = var_t

        tau: float = self._get_tau(
            var_t=var_t, s=similarity_score, t_hat=t_hat, metadata=metadata
        )

        u: float = random.uniform(0, 1)
        if u <= tau:
            return _Action.EXPLORE
        else:
            return _Action.EXPLOIT

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Optimize parameters with logistic regression
        Args
            similarities: np.ndarray - The similarities of the embeddings
            labels: np.ndarray - The labels of the embeddings
            metadata: EmbeddingMetadataObj - The metadata of the embedding
        Returns
            t_hat: float - The estimated threshold
            gamma: float - The estimated gamma
            var_t: float - The estimated variance of t
        """

        similarities = sm.add_constant(similarities)

        try:
            if len(similarities) != len(labels):
                print(f"len does not match: {len(similarities)} != {len(labels)}")
            self.logistic_regression.fit(similarities, labels)
            intercept, gamma = self.logistic_regression.coef_[0]

            gamma = max(gamma, 1e-6)
            t_hat = -intercept / gamma
            t_hat = float(np.clip(t_hat, 0.0, 1.0))

            similarities_col = (
                similarities[:, 1] if similarities.shape[1] > 1 else similarities[:, 0]
            )
            perfect_seperation = np.min(similarities_col[labels == 1]) > np.max(
                similarities_col[labels == 0]
            )
            var_t = self._get_var_t(
                perfect_seperation=perfect_seperation,
                n_observations=len(similarities),
                X=similarities,
                gamma=gamma,
                intercept=intercept,
            )

            return round(t_hat, 3), round(gamma, 3), var_t

        except Exception as e:
            print(f"Logistic regression failed: {e}")
            return -1.0, -1.0, -1.0

    def _get_var_t(
        self,
        perfect_seperation: bool,
        n_observations: int,
        X: np.ndarray,
        gamma: float,
        intercept: float,
    ) -> float:
        """
        Compute the variance of t using the delta method
        Args
            perfect_seperation: bool - Whether the data is perfectly separable
            n_observations: int - The number of observations
            X: np.ndarray - The design matrix
            gamma: float - The gamma parameter
            intercept: float - The intercept parameter
        Returns
            float - The variance of t
        Note:
            If the data is perfectly separable, we use the variance map to estimate the variance of t
            Otherwise, we use the delta method to estimate the variance of t
        """
        if perfect_seperation:
            if n_observations in self.variance_map:
                var_t = self.variance_map[n_observations]
            else:
                max_observations = max(self.variance_map.keys())
                var_t = self.variance_map[max_observations]
            return var_t
        else:
            p = self.logistic_regression.predict_proba(X)[:, 1]
            W = p * (1 - p)
            H = X.T @ (W[:, None] * X)

            cov_beta = np.linalg.inv(H)

            grad = np.array([-1.0 / gamma, intercept / (gamma**2)])

            var_t_hat = float(grad @ cov_beta @ grad)
            var_t_hat = max(0.0, var_t_hat)
            return var_t_hat

    def _get_tau(
        self,
        var_t: float,
        s: float,
        t_hat: float,
        metadata: EmbeddingMetadataObj,
    ) -> float:
        """
        Find the minimum tau value for the given similarity score
        Args
            var_t: float - The variance of t
            s: float - The similarity score between the query and the nearest neighbor
            t_hat: float - The estimated threshold
            metadata: EmbeddingMetadataObj - The metadata of the nearest neighbor
        Returns
            float - The minimum tau value
        """
        t_primes: List[float] = self._get_t_primes(t_hat=t_hat, var_t=var_t)
        likelihoods = self._likelihood(s=s, t=t_primes, gamma=metadata.gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods

        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        metadata.t_prime = t_primes[np.argmin(taus)]
        return round(np.min(taus), 5)

    def _get_t_primes(self, t_hat: float, var_t: float) -> List[float]:
        """
        Compute all possible t_prime values.
        Args
            t_hat: float - The estimated threshold
            var_t: float - The variance of t
        Returns
            List[float] - The t_prime values
        """
        t_primes: List[float] = np.array(
            [
                self._confidence_interval(
                    t_hat=t_hat, var_t=var_t, quantile=(1 - self.epsilon_grid[i])
                )
                for i in range(len(self.epsilon_grid))
            ]
        )
        return t_primes

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
        """
        Compute the likelihood of the given similarity score and threshold
        Args
            s: float - The similarity score between the query and the nearest neighbor
            t: float - The threshold
            gamma: float - The gamma parameter
        Returns
            float - The likelihood of the given similarity score and threshold
        """
        z = gamma * (s - t)
        return expit(z)
