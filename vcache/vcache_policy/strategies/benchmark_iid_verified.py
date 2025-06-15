from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm
from typing_extensions import override

from vcache.config import VCacheConfig
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy


class BenchmarkVerifiedIIDDecisionPolicy(VCachePolicy):
    """
    Policy that uses the vCache IID algorithm to compute optimal thresholds for each embedding.

    IMPORTANT: This policy is used for benchmark purposes and should not be used in production.
    """

    def __init__(
        self,
        similarity_evaluator: SimilarityEvaluator = StringComparisonSimilarityEvaluator(),
        delta: float = 0.01,
    ):
        """
        Initialize IID local threshold policy.

        Args:
            similarity_evaluator: The similarity evaluator to use for response comparison.
            delta: The delta value for the algorithm.
        """
        self.similarity_evaluator = similarity_evaluator
        self.bayesian = _Algorithm(delta=delta)
        self.inference_engine = None
        self.cache = None

    @override
    def setup(self, config: VCacheConfig):
        """
        Setup the policy with the given configuration.

        Args:
            config: The VCache configuration to use.
        """
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            vector_db=config.vector_db,
            eviction_policy=config.eviction_policy,
        )

    @override
    def process_request(
        self, prompt: str, system_prompt: Optional[str]
    ) -> tuple[bool, str, str]:
        """
        Process a request using IID local threshold policy.

        Args:
            prompt: The prompt to check for cache hit.
            system_prompt: The optional system prompt to use for the response. It will override the system prompt in the VCacheConfig if provided.

        Returns:
            Tuple containing [is_cache_hit, actual_response, nn_response].

        Raises:
            ValueError: If policy has not been setup.
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
                should_have_exploited = self.similarity_evaluator.answers_similar(
                    a=response, b=metadata.response
                )
                self.bayesian.update_metadata(
                    similarity_score=similarity_score,
                    is_correct=should_have_exploited,
                    metadata=metadata,
                )
                if not should_have_exploited:
                    self.cache.add(prompt=prompt, response=response)
                self.cache.update_metadata(
                    embedding_id=embedding_id, embedding_metadata=metadata
                )
                return False, response, metadata.response


class _Action(Enum):
    """
    Actions that can be taken by the IID algorithm.
    """

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Algorithm:
    """
    IID algorithm implementation for computing optimal thresholds.
    """

    def __init__(self, delta: float):
        """
        Initialize the IID algorithm.

        Args:
            delta: The delta parameter for the algorithm.
        """
        self.delta: float = delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.thold_grid: np.ndarray = np.linspace(0, 1, 20)

    def update_metadata(
        self, similarity_score: float, is_correct: bool, metadata: EmbeddingMetadataObj
    ) -> None:
        """
        Update the metadata with the new observation.

        Args:
            similarity_score: The similarity score between the query and the embedding.
            is_correct: Whether the query was correct.
            metadata: The metadata of the embedding.
        """
        if is_correct:
            metadata.observations.append((round(similarity_score, 3), 1))
        else:
            metadata.observations.append((round(similarity_score, 3), 0))

    def wilson_proportion_ci(self, cdf_estimates, n, confidence):
        """
        Vectorized Wilson score confidence interval for binomial proportions.

        Args:
            cdf_estimates: Array of CDF estimates (1,tholds,1).
            n: Number of trials (1).
            confidence: Confidence level for the interval (1,1,epsilons).

        Returns:
            Tuple of lower and upper bounds of the confidence interval.
        """
        k = np.asarray((cdf_estimates * n).astype(int))  # (1, tholds,1)
        n = np.asarray(n)  # 1

        assert np.all((0 <= k) & (k <= n)), "k must be between 0 and n"
        assert np.all(n > 0), "n must be > 0"

        p_hat = k / n  # (1, tholds,1)
        z = norm.ppf(confidence)  # this is single sided # (1,1,epsilons)

        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)) / denom

        ci_low = center - margin
        ci_upp = center + margin

        return ci_low, ci_upp  # (1,tholds,epsilons)

    def select_action(
        self, similarity_score: float, metadata: EmbeddingMetadataObj
    ) -> _Action:
        """
        Select the action to take based on the similarity score and observations.

        Args:
            similarity_score: The similarity score between the query and the embedding.
            metadata: The metadata of the embedding.

        Returns:
            The action to take (EXPLORE or EXPLOIT).
        """

        similarity_score = round(similarity_score, 3)
        similarities: np.ndarray = np.array([obs[0] for obs in metadata.observations])
        labels: np.ndarray = np.array([obs[1] for obs in metadata.observations])
        if len(similarities) < 6 or len(labels) < 6:
            return _Action.EXPLORE
        num_positive_samples = np.sum(labels == 1)
        num_negative_samples = np.sum(labels == 0)

        # ( for vectorization , [samples, tholds, epsilon])
        negative_samples = similarities[labels == 0].reshape(-1, 1, 1)
        labels = labels.reshape(-1, 1, 1)
        tholds = self.thold_grid.reshape(1, -1, 1)
        epsilon = self.epsilon_grid.reshape(1, 1, -1)

        cdf_estimate = (
            np.sum(negative_samples < tholds, axis=0, keepdims=True)
            / num_negative_samples
        )  # (1, tholds, 1)
        cdf_ci_lower, cdf_ci_upper = self.wilson_proportion_ci(
            cdf_estimate, num_negative_samples, confidence=1 - epsilon
        )  # (1, tholds, epsilon)

        pc_adjusted = (
            1
            - self.delta
            * (num_negative_samples + num_positive_samples)
            / num_negative_samples
        ) / (1 - epsilon)  # adjust for positive samples (1,1,epsilon)

        t_hats = (
            np.sum(cdf_estimate > pc_adjusted, axis=1, keepdims=True) == 0
        ) * 1.0 + (
            1 - (np.sum(cdf_estimate > pc_adjusted, axis=1, keepdims=True) == 0)
        ) * self.thold_grid[
            np.argmax(cdf_estimate > pc_adjusted, axis=1, keepdims=True)
        ]
        t_primes = (
            np.sum(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True) == 0
        ) * 1.0 + (
            1 - (np.sum(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True) == 0)
        ) * self.thold_grid[
            np.argmax(cdf_ci_lower > pc_adjusted, axis=1, keepdims=True)
        ]

        t_hat = np.min(t_hats)
        t_prime = np.min(t_primes)
        # if t_prime < 1.0:
        #     print(f"t_hat: {t_hat}, t_prime: {t_prime} num_positive_samples: {num_positive_samples} num_negative_samples: {num_negative_samples}")
        metadata.t_prime = t_prime
        metadata.t_hat = t_hat
        metadata.var_t = -1  # not computed

        if similarity_score <= t_prime:
            return _Action.EXPLORE
        else:
            return _Action.EXPLOIT
