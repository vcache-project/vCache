import unittest
from unittest.mock import MagicMock

from dotenv import load_dotenv

from vcache import (
    NoCachePolicy,
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
    )
    policy = NoCachePolicy()
    return config, policy


class TestvcacheNoCache(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that when cache is disabled, all requests are misses."""
        config, policy = create_default_config_and_policy()

        # Mock the cache attribute that main.py expects on the policy
        mock_cache = MagicMock()
        mock_cache.vector_db_size.return_value = 0
        policy.cache = mock_cache

        vcache = VCache(config, policy)

        # First request should be a miss
        cache_hit, response, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "Request should be a miss with cache disabled")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Same request should still be a miss with cache disabled
        cache_hit, response, _ = vcache.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(
            cache_hit, "Identical request should still be a miss with cache disabled"
        )
        self.assertTrue(len(response) > 0, "Response should not be empty")


if __name__ == "__main__":
    unittest.main()
