from typing import Optional

from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vectorq.vectorq_core.cache.embedding_engine import OpenAIEmbeddingEngine
from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vectorq.vectorq_core.cache.embedding_store.vector_db import VectorDB
from vectorq.vectorq_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vectorq.vectorq_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vectorq.vectorq_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy


class VectorQConfig:
    """
    VectorQConfig is a class that contains the configuration for the VectorQ system.
    It is used to configure the VectorQ system with the appropriate parameters.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine = OpenAIInferenceEngine(),
        embedding_engine: EmbeddingEngine = OpenAIEmbeddingEngine(),
        vector_db: VectorDB = HNSWLibVectorDB(),
        embedding_metadata_storage: EmbeddingMetadataStorage = InMemoryEmbeddingMetadataStorage(),
        eviction_policy: EvictionPolicy = LRUEvictionPolicy(),
        system_prompt: Optional[str] = None,
    ):
        self.inference_engine = inference_engine
        self.embedding_engine = embedding_engine
        self.vector_db = vector_db
        self.eviction_policy = eviction_policy
        self.embedding_metadata_storage = embedding_metadata_storage
        self.system_prompt = system_prompt
