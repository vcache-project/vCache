'''
Integration test 1
Test that the cache is working with 
    - Dummy inference engine (instead of OpenAI to avoid API dependencies)
    - OpenAI embedding engine
    - HNSW vector db
'''

import unittest
import asyncio
from vectorq.main import VectorQ
from vectorq.config import VectorQConfig
from vectorq.config import EmbeddingEngineType
from vectorq.config import InferenceEngineType
from vectorq.config import VectorDBType
from vectorq.config import SimilarityMetricType

class TestVectorDBStrategy(unittest.TestCase):
    
    def test_1(self):
        asyncio.run(self.async_test_1())
        
    async def async_test_1(self):
        config: VectorQConfig = VectorQConfig(
            enable_cache=True,
            is_static_threshold=False,
            
            inference_engine_type=InferenceEngineType.OPENAI,
            inference_engine_model_name="gpt-4o-mini",
            inference_engine_temperature=0.0,
            
            embedding_engine_type=EmbeddingEngineType.OPENAI,
            embedding_engine_model_name="text-embedding-ada-002",
            
            vector_db_type=VectorDBType.HNSW,
            vector_db_similarity_metric_type=SimilarityMetricType.COSINE,
        )
        
        vectorq: VectorQ = VectorQ(vectorq_config=config)
        
        response_1, cache_hit_1 = await vectorq.create(prompt="Is the sky blue?")
        response_2, cache_hit_2 = await vectorq.create(prompt="Is the grass green?")
        
        await vectorq.shutdown()
        
        self.assertTrue(len(response_1) > 0)
        self.assertTrue(len(response_2) > 0)
        self.assertFalse(cache_hit_1)
        self.assertFalse(cache_hit_2)
        
if __name__ == "__main__":
    unittest.main()
