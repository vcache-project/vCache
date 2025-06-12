import logging
import random
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.similarity_evaluator import (
    SimilarityEvaluator,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy


class DynamicLocalThresholdPolicy(VCachePolicy):
    """A policy that uses a dynamic, per-embedding threshold to make cache decisions.

    This policy implements the vCache algorithm, which uses a probabilistic approach
    to learn an optimal similarity threshold for each cached item. It balances
    exploiting the cache and exploring new responses to refine its decision boundaries.

    Attributes:
        bayesian (_Algorithm): The core algorithm for action selection and updates.
        similarity_evaluator (SimilarityEvaluator): Component for comparing responses.
        inference_engine (InferenceEngine): The LLM for generating new responses.
        cache (Cache): The vCache instance.
    """

    def __init__(self, delta: float = 0.01, max_background_workers: int = 100):
        """Initializes the policy.

        Args:
            delta (float): The desired error bound the cache needs to maintain.
            max_background_workers (int): Max threads for background processing.
        """
        self.bayesian = _Algorithm(delta=delta)
        self.similarity_evaluator: SimilarityEvaluator = None
        self.inference_engine: InferenceEngine = None
        self.cache: Cache = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_background_workers, thread_name_prefix="vcache-bg"
        )
        self._logger = logging.getLogger(__name__)

    @override
    def setup(self, config: VCacheConfig):
        """Configure the policy with the necessary components from VCacheConfig."""
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
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """Process a request to decide whether to serve from cache or generate a new response.

        This method finds the nearest neighbor in the cache. If none exists, it
        generates a new response. Otherwise, it uses the Bayesian algorithm to
        decide whether to EXPLOIT (use the cached response) or EXPLORE (generate a
        new one). In the EXPLORE case, label generation happens in the background.

        Args:
            prompt (str): The user's prompt.
            system_prompt (str, optional): An optional system prompt to guide the LLM.

        Returns:
            tuple[bool, str, str]: A tuple containing:
                - is_cache_hit (bool): True if the response is from the cache (EXPLOIT).
                - actual_response (str): The response served.
                - nn_response (str): The nearest neighbor's response, if one was found.
        """
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        knn = self.cache.get_knn(prompt=prompt, k=1)
        if not knn:
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, ""

        similarity_score, embedding_id = knn[0]
        metadata = self.cache.get_metadata(embedding_id=embedding_id)
        action = self.bayesian.select_action(
            similarity_score=similarity_score, metadata=metadata
        )

        match action:
            case _Action.EXPLOIT:
                return True, metadata.response, metadata.response
            case _Action.EXPLORE:
                response = self.inference_engine.create(
                    prompt=prompt, system_prompt=system_prompt
                )

                self._executor.submit(
                    self._generate_label,
                    response=response,
                    nn_response=metadata.response,
                    similarity_score=similarity_score,
                    embedding_id=embedding_id,
                    prompt=prompt,
                )

                return False, response, metadata.response

    def _generate_label(
        self,
        response: str,
        nn_response: str,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
    ):
        """Generate a label for a response and update metadata.

        This function runs in a background thread. It compares the newly generated
        response with the nearest neighbor's response to determine if the cache
        *should* have been hit. It then updates the metadata with this new
        observation and adds the new response to the cache if it was dissimilar.

        Args:
            response (str): The newly generated response.
            nn_response (str): The cached response of the nearest neighbor.
            similarity_score (float): The similarity between the query and the neighbor.
            embedding_id (int): The ID of the nearest neighbor embedding to update.
            prompt (str): The original prompt, to be cached if the new response is kept.
        """
        try:
            should_have_exploited = self.similarity_evaluator.answers_similar(
                a=response, b=nn_response
            )

            label: int = 1 if should_have_exploited else 0
            observation: Tuple[float, int] = (round(similarity_score, 3), label)
            self.cache.add_observation(
                embedding_id=embedding_id, observation=observation
            )

            if not should_have_exploited:
                self.cache.add(prompt=prompt, response=response)

        except Exception as e:
            self._logger.error(
                f"Error in background label generation: {e}", exc_info=True
            )

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and shutdown the thread pool."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)


class _Action(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Algorithm:
    """Implements the Bayesian algorithm for the DynamicLocalThresholdPolicy."""

    def __init__(self, delta: float):
        """Initializes the algorithm.

        Args:
            delta (float): The desired error bound the cache needs to maintain.
        """
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

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> _Action:
        """Select whether to EXPLORE or EXPLOIT based on the learned threshold.

        This method estimates the current threshold `t_hat` from observations,
        calculates a confidence-based exploration probability `tau`, and then
        randomly decides whether to explore or exploit.

        Args:
            similarity_score (float): The similarity of the current prompt to the cache entry.
            metadata (EmbeddingMetadataObj): The metadata of the cache entry.

        Returns:
            _Action: The action to perform, either EXPLORE or EXPLOIT.
        """
        similarity_score = round(similarity_score, 3)
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 6 or len(labels) < 6:
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
        """Estimate logistic regression parameters from observations.

        This method fits a logistic regression model to the similarity scores and
        labels to estimate the decision boundary parameters.

        Args:
            similarities (np.ndarray): The observed similarity scores.
            labels (np.ndarray): The observed labels (1 for correct, 0 for incorrect).

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - t_hat: The estimated similarity threshold.
                - gamma: The steepness of the logistic curve.
                - var_t: The variance of the threshold estimate.
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
        """Compute the variance of the threshold estimate `t_hat`.

        If the data is perfectly separable, it uses a pre-computed variance map.
        Otherwise, it uses the delta method to approximate the variance from the
        logistic regression's covariance matrix.

        Args:
            perfect_seperation (bool): True if the data is perfectly separable.
            n_observations (int): The number of observations.
            X (np.ndarray): The design matrix for the regression.
            gamma (float): The gamma parameter from the regression.
            intercept (float): The intercept from the regression.

        Returns:
            float: The variance of the threshold estimate.
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
        """Calculate the exploration probability `tau`.

        This method computes `tau`, the probability of choosing to EXPLORE. It's
        based on finding the worst-case (minimum) confidence that the current
        action is correct, considering the uncertainty in the threshold `t_hat`.

        Args:
            var_t (float): The variance of the threshold estimate.
            s (float): The similarity score of the current request.
            t_hat (float): The estimated threshold.
            metadata (EmbeddingMetadataObj): The metadata of the cache entry.

        Returns:
            float: The calculated exploration probability, `tau`.
        """
        t_primes: List[float] = self._get_t_primes(t_hat=t_hat, var_t=var_t)
        likelihoods = self._likelihood(s=s, t=t_primes, gamma=metadata.gamma)
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods

        taus = 1 - (1 - self.P_c) / (1 - alpha_lower_bounds)
        metadata.t_prime = t_primes[np.argmin(taus)]
        return round(np.min(taus), 5)

    def _get_t_primes(self, t_hat: float, var_t: float) -> List[float]:
        """Compute a grid of possible threshold values based on the confidence interval.

        Args:
            t_hat (float): The estimated threshold.
            var_t (float): The variance of the threshold estimate.

        Returns:
            List[float]: A list of potential threshold values (`t_prime`).
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
        """Calculate the upper bound of a confidence interval for the threshold `t`.

        This computes a threshold `t'` such that the estimated probability of the true
        threshold being greater than `t'` is less than or equal to `1 - quantile`.

        Args:
            t_hat (float): The estimated threshold.
            var_t (float): The variance of the threshold estimate.
            quantile (float): The desired quantile for the confidence interval.

        Returns:
            float: The upper bound of the confidence interval (`t_prime`).
        """
        z = norm.ppf(quantile)
        t_prime = t_hat + z * np.sqrt(var_t)
        return float(np.clip(t_prime, 0.0, 1.0))

    def _likelihood(self, s: float, t: float, gamma: float) -> float:
        """Compute the likelihood of a correct cache hit given a similarity score.

        This function uses the logistic (sigmoid) function to model the
        probability of a correct match based on the similarity `s`, a threshold
        `t`, and a steepness parameter `gamma`.

        Args:
            s (float): The similarity score.
            t (float): The decision threshold.
            gamma (float): The steepness of the logistic curve.

        Returns:
            float: The likelihood of a correct match.
        """
        z = gamma * (s - t)
        return expit(z)
