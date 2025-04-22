from typing import Optional

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vectorq.vectorq_core.cache.embedding_store.vector_db import VectorDB
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from vectorq.vectorq_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import EmbeddingMetadataStorage
from vectorq.vectorq_core.similarity_evaluator.strategies.string_comparison import StringComparisonSimilarityEvaluator
from vectorq.vectorq_core.vectorq_policy.vectorq_policy import VectorQPolicy

class VectorQConfig:
    '''
    VectorQConfig is a class that contains the configuration for the VectorQ system.
    It is used to configure the VectorQ system with the appropriate parameters.
    '''
    def __init__(
        self,
        accuracy_target: float = 0.8, 
        max_capacity: int = 1000, 
        enable_cache: bool = True, 
        rnd_num_ub: float = 1.0,
        is_static_threshold: bool = False,
        static_threshold: float = 0.0,

        inference_engine: Optional[InferenceEngine] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_db: Optional[VectorDB] = None,
        similarity_evaluator: SimilarityEvaluator = StringComparisonSimilarityEvaluator(),
        eviction_policy: Optional[EvictionPolicy] = None,
        embedding_metadata_storage: Optional[EmbeddingMetadataStorage] = None,
        vectorq_policy: Optional[VectorQPolicy] = None,
    ):
        self.accuracy_target: float = accuracy_target
        self.max_capacity: int = max_capacity
        self.enable_cache: bool = enable_cache
        self.rnd_num_ub: float = rnd_num_ub
        self.is_static_threshold: bool = is_static_threshold
        self.static_threshold: float = static_threshold
        
        self.inference_engine = inference_engine
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.similarity_evaluator = similarity_evaluator
        self.eviction_policy = eviction_policy
        self.embedding_metadata_storage = embedding_metadata_storage
        self.vectorq_policy = vectorq_policy
