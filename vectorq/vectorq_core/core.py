from typing import TYPE_CHECKING, List

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vectorq.vectorq_core.cache.embedding_store.embedding_store import EmbeddingStore

if TYPE_CHECKING:
    from vectorq.main import VectorQBenchmark
    from vectorq.config import VectorQConfig

from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.vectorq_policy.action import Action
class VectorQCore():
    
    def __init__(
            self, 
            vectorq_config: "VectorQConfig"
        ):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        self.inference_engine: "InferenceEngine" = self.vectorq_config.inference_engine
        self.similarity_evaluator: "SimilarityEvaluator" = self.vectorq_config.similarity_evaluator
        
        self.cache: Cache = Cache(
            embedding_store=EmbeddingStore(
                vectorq_config=self.vectorq_config
            ),
            embedding_engine=self.vectorq_config.embedding_engine,
            eviction_policy=self.vectorq_config.eviction_policy
        )
        self.vectorq_policy: VectorQPolicy = self.vectorq_config.vectorq_policy

    def process_request(self, prompt: str, benchmark: "VectorQBenchmark" = None, output_format: str = None) -> tuple[bool, str]:
        '''
        prompt: str - The prompt to check for cache hit
        benchmark: "VectorQBenchmark" - The optional benchmark object containing the pre-computed embedding and response
        output_format: str - The optional output format to use for the response
        Returns: tuple[bool, str] - Whether the cache hit or not and the response
        '''
        if (self.vectorq_config.is_static_threshold):
            return self.__naive(prompt, benchmark, output_format)
        else:
            return self.__vectorQ(prompt, benchmark, output_format)
        
    def __naive(self, prompt: str, benchmark: "VectorQBenchmark" = None, output_format: str = None) -> tuple[bool, str]:
        # TODO: LGS - Validate (Benchmark obj)
        if (self.cache.is_empty()):
            generated_response: str = benchmark.candidate_response
            self.cache.add_embedding(embedding=benchmark.candidate_embedding, response=generated_response)
            return False, generated_response
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(prompt, k=1, embedding=benchmark.candidate_embedding)
            sim, embedding_id = knn[0]
            if (sim >= self.vectorq_config.static_threshold):
                return True, self.cache.get_metadata(embedding_id=embedding_id).response
            else:
                embedding: List[float] = benchmark.candidate_embedding
                self.cache.add_embedding(embedding=embedding, response=generated_response)
                return False, benchmark.candidate_response
    
    def __vectorQ(self, prompt: str, benchmark: "VectorQBenchmark" = None, output_format: str = None) -> tuple[bool, str]:
        if (self.cache.is_empty()):
            response:str = self.inference_engine.create(prompt=prompt, output_format=output_format)
            self.cache.add(prompt=prompt, response=response)
            return False, response
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(prompt=prompt, k=1)
            similarity_score, embedding_id = knn[0]
            metadata_obj: EmbeddingMetadataObj = self.cache.get_metadata(embedding_id=embedding_id)
            
            selected_action: Action = self.vectorq_policy.select_action(similarity_score=similarity_score, metadata=metadata_obj)
            if (selected_action == Action.EXPLORE):
                response:str = self.inference_engine.create(prompt=prompt, output_format=output_format)
                should_have_exploited: bool = self.similarity_evaluator.answers_similar(a=response, b=benchmark.candidate_response)
                if (should_have_exploited):
                    self.vectorq_policy.update_policy(similarity_score=similarity_score, is_correct=True, metadata=metadata_obj)
                else:
                    self.vectorq_policy.update_policy(similarity_score=similarity_score, is_correct=False, metadata=metadata_obj)
                    self.cache.add(prompt=prompt, response=response)
                self.cache.update_metadata(embedding_id=embedding_id, metadata=metadata_obj)
                return False, response
            else:
                return True, metadata_obj.response
            
    def get_statistics(self) -> str:
        # TODO
        return ""
