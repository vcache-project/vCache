import unittest

from dotenv import load_dotenv

from vcache import (
    BenchmarkStaticDecisionPolicy,
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LangChainEmbeddingEngine,
    OpenAIInferenceEngine,
    VCache,
    VCacheConfig,
)

load_dotenv()


def create_default_config_and_policy():
    config = VCacheConfig(
        inference_engine=OpenAIInferenceEngine(
            model_name="gpt-4.1-nano-2025-04-14",
            temperature=0.0,
        ),
        embedding_engine=LangChainEmbeddingEngine(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        vector_db=HNSWLibVectorDB(),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
    )
    policy = BenchmarkStaticDecisionPolicy(threshold=0.8)
    return config, policy


class TestvcacheStaticThreshold(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the cache correctly identifies hits and misses."""
        config, policy = create_default_config_and_policy()
        vcache = VCache(config, policy)

        # First request should be a miss
        cache_hit, response, _, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "First request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Same request should be a hit
        cache_hit, response, _, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertTrue(cache_hit, "Identical request should be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Similar but different request, should be a hit
        cache_hit, response, _, _ = vcache.infer_with_cache_info(
            prompt="What's France's capital city?"
        )
        self.assertTrue(
            cache_hit, "Similar request should be a cache hit with high threshold"
        )
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Different request should be a miss
        cache_hit, response, _, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of Germany?"
        )
        self.assertFalse(cache_hit, "Different request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")

    def test_low_similarity_threshold(self):
        """Test different similarity thresholds effect on cache hits."""
        config, policy = create_default_config_and_policy()
        policy.threshold = 0.5
        vcache = VCache(config, policy)

        # First request should be a miss
        cache_hit, _, _, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "First request should be a cache miss")

        # Similar request with low threshold should be a hit
        cache_hit, _, _, _ = vcache.infer_with_cache_info(
            prompt="What's France's capital city?"
        )
        self.assertTrue(
            cache_hit, "Similar request should be a cache hit with low threshold"
        )

    def test_high_similarity_threshold(self):
        """Test different similarity thresholds effect on cache hits."""
        config, policy = create_default_config_and_policy()
        policy.threshold = 0.99
        vcache = VCache(config, policy)

        # First request should be a miss
        cache_hit, _, _, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "First request should be a cache miss")

        # Similar request with high threshold should be a miss
        cache_hit, _, _, _ = vcache.infer_with_cache_info(
            prompt="What's France's capital city?"
        )
        self.assertFalse(
            cache_hit, "Similar request should be a cache miss with high threshold"
        )


if __name__ == "__main__":
    unittest.main()
