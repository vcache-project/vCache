from typing import TYPE_CHECKING, List, Optional

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vectorq.vectorq_core.vectorq_policy.action import Action
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig


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
        system_prompt: Optional[str],
    ) -> tuple[bool, str, str]:
        """
        prompt: str - The prompt to check for cache hit
        output_format: str - The optional output format to use for the response
        Returns: tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response]
        """
        if self.cache.is_empty():
            response = self.inference_engine.create(
                prompt=prompt, system_prompt=system_prompt
            )
            self.cache.add(prompt=prompt, response=response)
            return False, response, ""
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(prompt=prompt, k=1)
            similarity_score, embedding_id = knn[0]
            metadata_obj = self.cache.get_metadata(embedding_id=embedding_id)
            selected_action = self.vectorq_policy.select_action(
                similarity_score=similarity_score, metadata=metadata_obj
            )

            match selected_action:
                case Action.REJECT:
                    response = self.inference_engine.create(
                        prompt=prompt, system_prompt=system_prompt
                    )
                    return False, response, ""
                case Action.EXPLORE:
                    response = self.inference_engine.create(
                        prompt=prompt, system_prompt=system_prompt
                    )
                    should_have_exploited = self.similarity_evaluator.answers_similar(
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
                        self.cache.add(prompt=prompt, response=response)
                    self.cache.update_metadata(
                        embedding_id=embedding_id, embedding_metadata=metadata_obj
                    )
                    return False, response, metadata_obj.response
                case Action.EXPLOIT:
                    return True, metadata_obj.response, metadata_obj.response

    def get_statistics(self) -> str:
        # TODO
        return ""
