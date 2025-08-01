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
    Configuration class that contains all settings for the vCache system.
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
        """
        Initialize VCache configuration with all necessary components.

        Args:
            inference_engine: Engine for generating responses from prompts.
            embedding_engine: Engine for generating embeddings from text.
            vector_db: Vector database for storing and retrieving embeddings.
            embedding_metadata_storage: Storage for embedding metadata.
            eviction_policy: Policy for removing items from cache when full.
            similarity_evaluator: Evaluator for determining similarity between prompts.
            system_prompt: Optional system prompt to use for all inferences.
        """
        self.inference_engine = inference_engine
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.eviction_policy = eviction_policy
        self.embedding_metadata_storage = embedding_metadata_storage
        self.similarity_evaluator = similarity_evaluator
        self.system_prompt = system_prompt
