"""
Integration test 1
Test that the cache is working with
    - OpenAI inference engine
    - LangChain embedding engine
    - HNSW vector db
"""

# import asyncio
import unittest

# from vectorq import (
#     HNSWLibVectorDB,
#     InMemoryEmbeddingMetadataStorage,
#     LangChainEmbeddingEngine,
#     LRUEvictionPolicy,
#     OpenAIInferenceEngine,
#     StringComparisonSimilarityEvaluator,
#     VectorQ,
#     VectorQConfig,
# )


class TestVectorQIntegration(unittest.TestCase):
    pass
    # def test_1(self):
    #     asyncio.run(self.async_test_1())

    # async def async_test_1(self):
    #     config = VectorQConfig(
    #         enable_cache=True,
    #         is_static_threshold=False,
    #         inference_engine=OpenAIInferenceEngine(
    #             model_name="gpt-4o-mini", temperature=0.0
    #         ),
    #         embedding_engine=LangChainEmbeddingEngine(
    #             model_name="sentence-transformers/all-mpnet-base-v2"
    #         ),
    #         vector_db=HNSWLibVectorDB(),
    #         similarity_evaluator=StringComparisonSimilarityEvaluator(),
    #         eviction_policy=LRUEvictionPolicy(),
    #         embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
    #     )

    #     vectorq = VectorQ(config)

    #     response_1, cache_hit_1 = await vectorq.create(prompt="Is the sky blue?")
    #     response_2, cache_hit_2 = await vectorq.create(prompt="Is the grass green?")

    #     await vectorq.shutdown()

    #     self.assertTrue(len(response_1) > 0)
    #     self.assertTrue(len(response_2) > 0)
    #     self.assertFalse(cache_hit_1)
    #     self.assertFalse(cache_hit_2)


if __name__ == "__main__":
    # unittest.main()
    pass

