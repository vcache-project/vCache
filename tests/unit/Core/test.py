import asyncio
import unittest
from unittest.mock import MagicMock

from vectorq.config import VectorQConfig
from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.main import VectorQ
from vectorq.vectorq_core.core import VectorQCore


class TestVectorQAsyncQueue(unittest.TestCase):
    def setUp(self):
        self.mock_inference_engine = MagicMock(spec=InferenceEngine)

        self.mock_core = MagicMock(spec=VectorQCore)

        self.config_with_cache = VectorQConfig(
            enable_cache=True, inference_engine=self.mock_inference_engine
        )
        self.config_without_cache = VectorQConfig(
            enable_cache=False, inference_engine=self.mock_inference_engine
        )

    def test_multiple_concurrent_requests(self):
        # Set up mock responses
        self.mock_inference_engine.create.side_effect = [
            "response 1",
            "response 2",
            "response 3",
            "response 4",
            "response 5",
        ]

        async def test():
            vectorq = VectorQ(self.config_without_cache)
            tasks = [vectorq.create(f"prompt {i}") for i in range(5)]
            results = await asyncio.gather(*tasks)
            await vectorq.shutdown()
            return results

        results = asyncio.run(test())

        self.assertEqual(len(results), 5)
        for i, (response, cache_hit) in enumerate(results):
            self.assertEqual(response, f"response {i + 1}")
            self.assertFalse(cache_hit)

        self.assertEqual(self.mock_inference_engine.create.call_count, 5)

    def test_error_handling(self):
        # Set up mock to raise an exception
        self.mock_inference_engine.create.side_effect = Exception("Test error")

        async def test():
            vectorq = VectorQ(self.config_without_cache)
            response, cache_hit = await vectorq.create("Is the sky blue?")
            await vectorq.shutdown()
            return response, cache_hit

        response, cache_hit = asyncio.run(test())

        self.assertTrue(response.startswith("[ERROR]"))
        self.assertFalse(cache_hit)

    def test_fifo_processing_order(self):
        processing_times = []

        def create_side_effect(prompt, output_format=None):
            processing_times.append(prompt)
            return f"response for {prompt}"

        self.mock_inference_engine.create.side_effect = create_side_effect

        async def test():
            vectorq = VectorQ(self.config_without_cache)

            tasks = [
                vectorq.create("prompt 1"),
                vectorq.create("prompt 2"),
                vectorq.create("prompt 3"),
            ]

            results = await asyncio.gather(*tasks)

            await vectorq.shutdown()
            return results

        _ = asyncio.run(test())

        self.assertEqual(processing_times, ["prompt 1", "prompt 2", "prompt 3"])


if __name__ == "__main__":
    unittest.main()
