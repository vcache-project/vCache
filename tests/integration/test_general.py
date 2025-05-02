import unittest

from dotenv import load_dotenv

from vectorq import (
    LangChainEmbeddingEngine,
    OpenAIInferenceEngine,
    VectorQ,
    VectorQConfig,
)

load_dotenv()


class TestVectorQGeneral(unittest.TestCase):
    def test_cache_disabled(self):
        """Test that when cache is disabled, all requests are misses."""
        config = VectorQConfig(
            enable_cache=False,
            inference_engine=OpenAIInferenceEngine(
                model_name="gpt-4o-mini", temperature=0.0
            ),
            embedding_engine=LangChainEmbeddingEngine(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
        )

        vectorq = VectorQ(config)

        # First request should be a miss
        cache_hit, response, _ = vectorq.create(prompt="What is the capital of France?")
        self.assertFalse(cache_hit, "Request should be a miss with cache disabled")
        self.assertTrue(len(response) > 0, "Response should not be empty")

        # Same request should still be a miss with cache disabled
        cache_hit, response, _ = vectorq.create(prompt="What is the capital of France?")
        self.assertFalse(
            cache_hit, "Identical request should still be a miss with cache disabled"
        )
        self.assertTrue(len(response) > 0, "Response should not be empty")


if __name__ == "__main__":
    unittest.main()
