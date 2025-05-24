from typing import Optional

from vcache.inference_engine.inference_engine import InferenceEngine
from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.vcache_core.cache.embedding_engine import OpenAIEmbeddingEngine
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import VectorDB
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)


class VCacheConfig:
    """
    VCacheConfig is a class that contains the configuration for the vCache system.
    It is used to configure the vCache system with the appropriate parameters.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine = OpenAIInferenceEngine(),
        embedding_engine: EmbeddingEngine = OpenAIEmbeddingEngine(),
        vector_db: VectorDB = HNSWLibVectorDB(),
        embedding_metadata_storage: EmbeddingMetadataStorage = InMemoryEmbeddingMetadataStorage(),
        eviction_policy: EvictionPolicy = NoEvictionPolicy(),
        similarity_evaluator: SimilarityEvaluator = StringComparisonSimilarityEvaluator(),
        system_prompt: Optional[str] = None,
    ):
        self.inference_engine = inference_engine
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.eviction_policy = eviction_policy
        self.embedding_metadata_storage = embedding_metadata_storage
        self.similarity_evaluator = similarity_evaluator
        self.similarity_evaluator.set_inference_engine(self.inference_engine)
        self.system_prompt = system_prompt
