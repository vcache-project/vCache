import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.main import VectorQ
    from vectorq.main import VectorQBenchmark
    from vectorq.inference_engine.inference_engine import InferenceEngine
    from vectorq.config import VectorQConfig
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.cache import Cache
from vectorq.vectorq_core.bayesian_inference.bayesian_inference import BayesianInference
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator

class VectorQCore():
    
    def __init__(
            self, 
            vectorq: "VectorQ", 
            inference_engine: InferenceEngine,
            vectorq_config: "VectorQConfig"
        ):
        self.vectorq: "VectorQ" = vectorq
        self.inference_engine: "InferenceEngine" = inference_engine
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.cache: "Cache" = Cache(vectorq_config=self.vectorq_config)
        self.bayesian_inference: "BayesianInference" = BayesianInference(vectorq_config=self.vectorq_config)
        self.similarity_evaluator: "SimilarityEvaluator" = SimilarityEvaluator(vectorq_config=self.vectorq_config)
        
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
        # TODO: LGS - Validate
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
        # TODO: LGS - Implement  
        if (self.cache.is_empty()):
            response:str = self.inference_engine.create(prompt=prompt, output_format=output_format)
            self.cache.add(prompt=prompt, response=response)
            return False, response
        else:
            knn: List[tuple[float, int]] = self.cache.get_knn(prompt=prompt, k=1)
            sim, embedding_id = knn[0]
            metadata: "EmbeddingMetadataObj" = self.cache.get_metadata(embedding_id=embedding_id)
            
            # TODO: LGS
            response:str = self.inference_engine.create(prompt=prompt, output_format=output_format)
            self.cache.add(prompt=prompt, response=response)
            return False, response
            
    def get_statistics(self) -> str:
        # TODO
        return ""
