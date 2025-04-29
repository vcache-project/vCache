from typing import TYPE_CHECKING, List, Optional

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    from vectorq.main import VectorQBenchmark


class VectorQCore:
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        self.inference_engine: "InferenceEngine" = self.vectorq_config.inference_engine
        self.similarity_evaluator: "SimilarityEvaluator" = (
            self.vectorq_config.similarity_evaluator
        )

        self.cache: Cache = Cache(
            embedding_store=EmbeddingStore(vectorq_config=self.vectorq_config),
            embedding_engine=self.vectorq_config.embedding_engine,
            eviction_policy=self.vectorq_config.eviction_policy,
        )
        self.vectorq_policy: VectorQPolicy = self.vectorq_config.vectorq_policy

    def process_request(
        self,
        prompt: str,
        benchmark: Optional["VectorQBenchmark"],
        output_format: Optional[str],
    ) -> tuple[bool, str]:
        """
        prompt: str - The prompt to check for cache hit
        benchmark: "VectorQBenchmark" - The optional benchmark object containing the pre-computed embedding and response
        output_format: str - The optional output format to use for the response
        Returns: tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        if self.vectorq_config.is_static_threshold:
            return self.__naive(prompt, benchmark, output_format)
        else:
            return self.__vectorQ(prompt, benchmark, output_format)

    def __naive(
        self,
        prompt: str,
        benchmark: Optional["VectorQBenchmark"],
        output_format: Optional[str],
    ) -> tuple[bool, str]:
        if self.cache.is_empty():
            response: str = self.__create(
                prompt=prompt, benchmark=benchmark, output_format=output_format
            )
            self.__add(prompt=prompt, response=response, benchmark=benchmark)
            return False, response, ""
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(
                prompt=prompt, k=1, benchmark=benchmark
            )
            sim, embedding_id = knn[0]
            metadata_obj: EmbeddingMetadataObj = self.cache.get_metadata(
                embedding_id=embedding_id
            )
            if sim >= self.vectorq_config.static_threshold:
                return True, metadata_obj.response, metadata_obj.response
            else:
                response: str = self.__create(
                    prompt=prompt, benchmark=benchmark, output_format=output_format
                )
                self.__add(prompt=prompt, response=response, benchmark=benchmark)
                return False, response, metadata_obj.response

    def __vectorQ(
        self,
        prompt: str,
        benchmark: Optional["VectorQBenchmark"],
        output_format: Optional[str],
    ) -> tuple[bool, str, str]:
        if self.cache.is_empty():
            response: str = self.__create(
                prompt=prompt, benchmark=benchmark, output_format=output_format
            )
            self.__add(prompt=prompt, response=response, benchmark=benchmark)
            return False, response, ""
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(
                prompt=prompt, k=1, benchmark=benchmark
            )
            similarity_score, embedding_id = knn[0]
            metadata_obj: EmbeddingMetadataObj = self.cache.get_metadata(
                embedding_id=embedding_id
            )
            selected_action: Action = self.vectorq_policy.select_action(
                similarity_score=similarity_score, metadata=metadata_obj
            )

            if selected_action == Action.EXPLORE:
                response: str = self.__create(
                    prompt=prompt, benchmark=benchmark, output_format=output_format
                )
                should_have_exploited: bool = self.similarity_evaluator.answers_similar(
                    a=response, b=metadata_obj.response
                )
                if should_have_exploited:
                    # TODO: Update policy directly in metadata object
                    self.vectorq_policy.update_policy(
                        similarity_score=similarity_score,
                        is_correct=True,
                        metadata=metadata_obj,
                    )
                else:
                    # TODO: Update policy directly in metadata object
                    self.vectorq_policy.update_policy(
                        similarity_score=similarity_score,
                        is_correct=False,
                        metadata=metadata_obj,
                    )
                    self.__add(prompt=prompt, response=response, benchmark=benchmark)
                self.cache.update_metadata(
                    embedding_id=embedding_id, embedding_metadata=metadata_obj
                )
                return False, response, metadata_obj.response
            else:
                return True, metadata_obj.response, metadata_obj.response

    def __create(
        self,
        prompt: str,
        benchmark: Optional["VectorQBenchmark"],
        output_format: Optional[str],
    ) -> str:
        if benchmark is not None:
            return benchmark.candidate_response
        else:
            return self.inference_engine.create(
                prompt=prompt, output_format=output_format
            )

    def __add(
        self, prompt: str, response: str, benchmark: Optional["VectorQBenchmark"]
    ) -> int:
        if benchmark is not None:
            return self.cache.add_embedding(
                embedding=benchmark.candidate_embedding, response=response
            )
        else:
            return self.cache.add(prompt=prompt, response=response)

    def get_statistics(self) -> str:
        # TODO
        return ""
