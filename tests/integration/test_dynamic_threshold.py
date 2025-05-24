import unittest

from dotenv import load_dotenv

from vcache import (
    DynamicLocalThresholdPolicy,
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LangChainEmbeddingEngine,
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
        embedding_engine=LangChainEmbeddingEngine(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        vector_db=HNSWLibVectorDB(),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        system_prompt="Please answer in a single word with the first letter capitalized. Example: London",
    )
    policy = DynamicLocalThresholdPolicy(delta=0.05)
    return config, policy


class TestVectorQDynamicThreshold(unittest.TestCase):
    def test_basic_functionality(self):
        """Test that the cache correctly identifies hits and misses."""
        config, policy = create_default_config_and_policy()
        vectorq = VectorQ(config, policy)

        # First request should be a miss
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="What is the capital of France?"
        )
        self.assertFalse(cache_hit, "First request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # The 2nd to 5th request should be miss because it's still adjusting the threshold
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="What's France's capital city?"
        )
        self.assertFalse(cache_hit, "Second request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="France's capital city is called what?"
        )
        self.assertFalse(cache_hit, "Identical request should be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="Tell me the capital city of France"
        )
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="Which city is the capital of France?"
        )

        # After several tries with the Bayesian policy, we should now get a hit
        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="The capital of France is?"
        )
        self.assertTrue(cache_hit, "Similar request should now be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        cache_hit, response, _ = vectorq.infer_with_cache_info(
            prompt="Can you tell me what the capital of France is?"
        )
        self.assertTrue(cache_hit, "Similar request should now be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")

    def test_high_delta(self):
        # TODO: Implement this
        self.assertTrue(True)

    def test_low_delta(self):
        # TODO: Implement this
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
