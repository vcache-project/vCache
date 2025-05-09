from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from typing_extensions import override

from vectorq.config import VectorQConfig
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore
from vectorq.vectorq_core.similarity_evaluator import (
    SimilarityEvaluator,
    StringComparisonSimilarityEvaluator,
)
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy


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

        if len(similarities) < 6 or len(labels) < 6:
            return _Action.EXPLORE

        t_primes: List[Tuple[float, float, float]] = np.array(
            [
                self._estimate_parameters(
                    similarities=similarities,
                    labels=labels,
                    epsilon=self.epsilon_grid[i],
                )
                for i in range(len(self.epsilon_grid))
            ]
        )

        t_prime_values = np.array([t[0] for t in t_primes])
        min_index = np.argmin(t_prime_values)
        t_prime, t_hat, var_t_hat = t_primes[min_index]

        metadata.t_prime = t_prime
        metadata.t_hat = t_hat
        metadata.var_t = var_t_hat

        if similarity_score <= t_prime:
            return _Action.EXPLORE
        else:
            return _Action.EXPLOIT

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray, epsilon: float
    ) -> Tuple[float, float, float]:
        """
        Compute the threshold under an IID assumption
        Args
            similarities: np.ndarray - The nearest neighbor similarity observations
            labels: np.ndarray - The nearest neighbor label observations
        Returns
            t_prime: float - The estimated threshold
            t_hat: float - The estimated threshold
            var_t_hat: float - The variance of t
        """

        n = len(labels)
        num_steps = 64

        try:
            # 1) Approximate t_hat
            thresholds: np.ndarray = np.linspace(
                min(similarities), max(similarities), num_steps
            )
            failures: np.ndarray = np.array(
                [np.sum((labels == 0) & (similarities > t)) for t in thresholds]
            )
            failure_rates: np.ndarray = failures / n

            delta_prime: float = (1 - self.delta) / (1 - epsilon)
            valid: np.ndarray = np.where(failure_rates <= (1 - delta_prime))[0]
            idx: int = valid[0] if valid.size > 0 else num_steps - 1
            t_hat: float = thresholds[idx]

            # 2) Approximate variance of t_hat
            f_t: float = self._approximate_f_t(
                idx=idx,
                num_steps=num_steps,
                thresholds=thresholds,
                failure_rates=failure_rates,
            )
            var_F_hat: float = self.delta * (1 - self.delta) / n
            var_t_hat: float = var_F_hat / (f_t**2) if f_t != 0 else np.inf

            # 3) Calculate t_prime
            z: float = norm.ppf(1 - epsilon)
            t_prime: float = t_hat + z * np.sqrt(var_t_hat)
            return float(np.clip(t_prime, 0.0, 1.0)), t_hat, var_t_hat

        except Exception as e:
            print(f"IID-based threshold estimation failed: {e}")
            return 1.0, -1, -1

    def _approximate_f_t(
        self,
        idx: int,
        num_steps: int,
        thresholds: np.ndarray,
        failure_rates: np.ndarray,
    ) -> float:
        """
        Approximate the failure rate at t_hat
        Args
            idx: int - The index of t_hat
            num_steps: int - The number of steps
            thresholds: np.ndarray - The thresholds
            failure_rates: np.ndarray - The failure rates
        Returns
            f_t: float - The failure rate at t_hat
        """
        if 0 < idx < num_steps - 1:
            dt: float = thresholds[idx + 1] - thresholds[idx - 1]
            dF: float = failure_rates[idx + 1] - failure_rates[idx - 1]
            f_t: float = -dF / dt
        else:
            if idx == 0:
                dt: float = thresholds[1] - thresholds[0]
                dF: float = failure_rates[1] - failure_rates[0]
            else:
                dt: float = thresholds[-1] - thresholds[-2]
                dF: float = failure_rates[-1] - failure_rates[-2]
            f_t: float = -dF / dt
        return f_t
