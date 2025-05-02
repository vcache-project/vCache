import unittest

from dotenv import load_dotenv

from vectorq import (
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LangChainEmbeddingEngine,
    OpenAIInferenceEngine,
    StringComparisonSimilarityEvaluator,
    VectorQ,
    VectorQBayesianPolicy,
    VectorQConfig,
)

load_dotenv()


def create_config():
    return VectorQConfig(
        inference_engine=OpenAIInferenceEngine(
            model_name="gpt-4.1-nano-2025-04-14", temperature=0.0
        ),
        embedding_engine=LangChainEmbeddingEngine(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        similarity_evaluator=StringComparisonSimilarityEvaluator(),
        vectorq_policy=VectorQBayesianPolicy(delta=0.05),
        vector_db=HNSWLibVectorDB(),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        system_prompt="Please answer in a single word with the first letter capitalized. Example: London",
    )


class TestVectorQDynamicThreshold(unittest.TestCase):
    def test_functionality(self):
        """Test that the cache correctly identifies hits and misses."""
        config = create_config()
        vectorq = VectorQ(config)

        # First request should be a miss
        cache_hit, response, _ = vectorq.create(prompt="What is the capital of France?")
        self.assertFalse(cache_hit, "First request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # The 2nd to 5th request should be miss because it's still adjusting the threshold
        cache_hit, response, _ = vectorq.create(prompt="What's France's capital city?")
        self.assertFalse(cache_hit, "Second request should be a cache miss")
        self.assertTrue(len(response) > 0, "Response should not be empty")
        cache_hit, response, _ = vectorq.create(
            prompt="France's capital city is called what?"
        )
        self.assertFalse(cache_hit, "Identical request should be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")
        cache_hit, response, _ = vectorq.create(
            prompt="Tell me the capital city of France"
        )
        cache_hit, response, _ = vectorq.create(
            prompt="Which city is the capital of France?"
        )

        # After several tries with the Bayesian policy, we should now get a hit
        cache_hit, response, _ = vectorq.create(prompt="The capital of France is?")
        self.assertTrue(cache_hit, "Similar request should now be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        cache_hit, response, _ = vectorq.create(
            prompt="Can you tell me what the capital of France is?"
        )
        self.assertTrue(cache_hit, "Similar request should now be a cache hit")
        self.assertTrue(len(response) > 0, "Response should not be empty")


if __name__ == "__main__":
    unittest.main()
