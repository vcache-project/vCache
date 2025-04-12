from typing import Optional, List
from vectorq.inference_engine.inference_engine import InferenceEngineType
from vectorq.vectorq_core.bayesian_inference.bayesian_inference import LikelihoodFunctionType
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluatorType
from vectorq.vectorq_core.cache.vector_db.vector_db import SimilarityMetricType
from vectorq.vectorq_core.cache.vector_db.vector_db import VectorDBType
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_storage import EmbeddingMetadataStorageType
from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngineType
from vectorq.vectorq_core.cache.eviction_policy.eviction_policy import EvictionPolicyType


class VectorQConfig:
    '''
    VectorQConfig is a class that contains the configuration for the VectorQ system.
    It is used to configure the VectorQ system with the appropriate parameters.
    It has default values for all the parameters.
    '''
    def __init__(
        self,
        accuracy_target: float = 0.8, 
        max_capacity: int = 1000, 
        enable_cache: bool = True, 
        rnd_num_ub: float = 1.0,
        is_static_threshold: bool = False,
        static_threshold: float = 0.0,
        
        inference_engine_type: InferenceEngineType = InferenceEngineType.OPENAI,
        inference_engine_model_name: str = "gpt-4o-mini",
        inference_engine_temperature: int = 0,
        
        embedding_metadata_storage_type: EmbeddingMetadataStorageType = EmbeddingMetadataStorageType.LANGCHAIN,
        
        embedding_engine_type: EmbeddingEngineType = EmbeddingEngineType.OPENAI,
        embedding_engine_model_name: str = "text-embedding-ada-002",
        
        vector_db_type: VectorDBType = VectorDBType.HNSW,
        vector_db_similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        
        likelihood_function_type: LikelihoodFunctionType = LikelihoodFunctionType.SIGMOID,
        
        similarity_evaluator_type: SimilarityEvaluatorType = SimilarityEvaluatorType.STRING_COMPARISON,
        
        eviction_policy_type: EvictionPolicyType = EvictionPolicyType.NONE
    ):
        self.accuracy_target: float = accuracy_target
        self.max_capacity: int = max_capacity
        self.enable_cache: bool = enable_cache
        self.rnd_num_ub: float = rnd_num_ub
        self.is_static_threshold: bool = is_static_threshold
        self.static_threshold: float = static_threshold
        self._inference_engine_type: InferenceEngineType = inference_engine_type
        self._inference_engine_model_name: str = inference_engine_model_name
        self._inference_engine_temperature: int = inference_engine_temperature
        self._embedding_metadata_storage_type: EmbeddingMetadataStorageType = embedding_metadata_storage_type
        self._embedding_engine_type: EmbeddingEngineType = embedding_engine_type
        self._embedding_engine_model_name: str = embedding_engine_model_name
        self._vector_db_type: VectorDBType = vector_db_type
        self._vector_db_similarity_metric_type: SimilarityMetricType = vector_db_similarity_metric_type
        self._likelihood_function_type: LikelihoodFunctionType = likelihood_function_type
        self._similarity_evaluator_type: SimilarityEvaluatorType = similarity_evaluator_type
        self._eviction_policy_type: EvictionPolicyType = eviction_policy_type
    