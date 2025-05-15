import unittest

from dotenv import load_dotenv

from vectorq import (
    NoCachePolicy,
    OpenAIInferenceEngine,
    VectorQ,
    VectorQConfig,
)

load_dotenv()


def create_default_config_and_policy():
    config = VectorQConfig(
        inference_engine=OpenAIInferenceEngine(
            model_name="gpt-4.1-nano-2025-04-14",
            temperature=0.0,
        ),
    )
    policy = NoCachePolicy()
    return config, policy


class TestVectorQNoCache(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that when cache is disabled, all requests are misses."""
        config, policy = create_default_config_and_policy()
        vectorq = VectorQ(config, policy)

        # First request should be a miss
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "Request should be a miss with cache disabled")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Same request should still be a miss with cache disabled
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(
            cache_hit, "Identical request should still be a miss with cache disabled"
        )
        self.assertTrue(len(response) > 0, "Response should not be empty")


if __name__ == "__main__":
    unittest.main()
