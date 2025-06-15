from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.open_ai import (
    OpenAIEmbeddingEngine,
)
from vcache.vcache_core.cache.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.verified import (
    VerifiedDecisionPolicy,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy

vcache_policy: VCachePolicy = VerifiedDecisionPolicy(delta=0.02)
vcache_config: VCacheConfig = VCacheConfig(
    inference_engine=OpenAIInferenceEngine(),
    embedding_engine=OpenAIEmbeddingEngine(),
    vector_db=HNSWLibVectorDB(
        similarity_metric_type=SimilarityMetricType.COSINE,
        max_capacity=100000,
    ),
    similarity_evaluator=StringComparisonSimilarityEvaluator,
)
vcache: VCache = VCache(vcache_config, vcache_policy)
