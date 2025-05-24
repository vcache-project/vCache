from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm
from typing_extensions import override

from vcache.config import VectorQConfig
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.vectorq_policy import VectorQPolicy


class IIDLocalThresholdPolicy(VectorQPolicy):
    def __init__(
        self,
        similarity_evaluator: SimilarityEvaluator = StringComparisonSimilarityEvaluator(),
        delta: float = 0.01,
    ):
        """
        This policy uses the VectorQ IID algorithm to compute the optimal threshold for each
        embedding in the cache.
        Each threshold is used to determine if a response is a cache hit.

        Args
            similarity_evaluator: SimilarityEvaluator - The similarity evaluator to use
            delta: float - The delta value to use
        """
        self.similarity_evaluator = similarity_evaluator
        self.bayesian = _Algorithm(delta=delta)
        self.inference_engine = None
        self.cache = None

    @override
    def setup(self, config: VectorQConfig):
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
        """
        Args
            prompt: str - The prompt to check for cache hit
            system_prompt: Optional[str] - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        Returns
            tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
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
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class _Algorithm:
    def __init__(self, delta: float):
        self.delta: float = delta
        self.epsilon_grid: np.ndarray = np.linspace(1e-6, 1 - 1e-6, 50)
        self.thold_grid: np.ndarray = np.linspace(0, 1, 20)

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

    def wilson_proportion_ci(self, cdf_estimates, n, confidence):
        """
        Vectorized Wilson score confidence interval for binomial proportions.

        Parameters:
        - k : array_like, number of successes (1,tholds,1)
        - n : array_like, number of trials (1)
        - confidence_level : float, confidence level for the interval (1,1,epsilons)

        Returns:
        - ci_low, ci_upp : np.ndarray, lower and upper bounds of the confidence interval
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
