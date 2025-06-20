import time
import unittest
from unittest.mock import patch

from dotenv import load_dotenv

from vcache import VCache, VCacheConfig, VerifiedDecisionPolicy
from vcache.vcache_core.cache.embedding_engine.strategies.lang_chain import (
    LangChainEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy

load_dotenv()


class TestEvictionPolicy(unittest.TestCase):
    def setUp(self):
        """Set up a VCache instance with a real LRU eviction policy."""
        self.max_size = 10
        self.watermark = 0.8
        self.eviction_percentage = 0.2
        self.num_to_evict = int(self.max_size * self.eviction_percentage)
        self.watermark_limit = int(self.max_size * self.watermark)

        eviction_policy = LRUEvictionPolicy(
            max_size=self.max_size,
            watermark=self.watermark,
            eviction_percentage=self.eviction_percentage,
        )

        config = VCacheConfig(
            embedding_engine=LangChainEmbeddingEngine(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
            vector_db=HNSWLibVectorDB(),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            eviction_policy=eviction_policy,
        )

        self.policy = VerifiedDecisionPolicy(delta=0.8)
        self.vcache = VCache(config, self.policy)
        self.vcache.vcache_policy.setup(config)

        # To test eviction, we must predictably fill the cache.
        # We patch get_knn to prevent cache hits, ensuring every .infer() call
        # adds a new item to the cache.
        self.patcher = patch.object(
            self.vcache.vcache_policy.cache, "get_knn", return_value=[]
        )
        self.patcher.start()

    def tearDown(self):
        """Clean up resources."""
        self.patcher.stop()
        self.policy.shutdown()
        self.vcache.vcache_config.eviction_policy.shutdown()

    def test_cache_fills_but_not_evict_before_watermark(self):
        """
        Verify that items are added to the cache and eviction is NOT
        triggered before the watermark is reached.
        """
        # Fill the cache to just below the watermark limit
        for i in range(self.watermark_limit - 1):
            prompt = f"prompt {i}"
            with patch.object(
                self.vcache.vcache_policy.inference_engine,
                "create",
                return_value=f"Response to '{prompt}'",
            ):
                self.vcache.infer(prompt=prompt)

            self.assertEqual(self.vcache.vcache_policy.cache.vector_db_size(), i + 1)

        # At this point, the cache size should be one less than the watermark limit
        # and no eviction should have occurred.
        self.assertEqual(
            self.vcache.vcache_policy.cache.vector_db_size(),
            self.watermark_limit - 1,
        )

    def test_eviction_triggers_at_watermark(self):
        """
        Confirm that the eviction process runs automatically when the
        cache size hits the watermark.
        """
        # Fill the cache to just below the watermark
        for i in range(self.watermark_limit - 1):
            prompt = f"prompt {i}"
            with patch.object(
                self.vcache.vcache_policy.inference_engine,
                "create",
                return_value=f"Response to '{prompt}'",
            ):
                self.vcache.infer(prompt=prompt)

        # This call should push the cache size to the watermark and trigger eviction
        trigger_prompt = "prompt trigger"
        with patch.object(
            self.vcache.vcache_policy.inference_engine,
            "create",
            return_value=f"Response to '{trigger_prompt}'",
        ):
            self.vcache.infer(prompt=trigger_prompt)

        time.sleep(1)  # Allow time for background eviction to complete

        # The size should be the watermark limit minus the number of evicted items
        expected_size = self.watermark_limit - self.num_to_evict
        self.assertEqual(
            self.vcache.vcache_policy.cache.vector_db_size(), expected_size
        )

    def test_lru_evicts_correct_items(self):
        """
        Verify that the LRU policy evicts the oldest items.
        """
        prompts = [f"prompt {i}" for i in range(self.watermark_limit + 1)]
        for i, prompt in enumerate(prompts):
            with patch.object(
                self.vcache.vcache_policy.inference_engine,
                "create",
                return_value=f"Response to '{prompt}'",
            ):
                self.vcache.infer(prompt=prompt)
            time.sleep(0.01)

        time.sleep(1)

        all_metadata = (
            self.vcache.vcache_policy.cache.get_all_embedding_metadata_objects()
        )
        remaining_responses = {m.response for m in all_metadata}

        for i in range(self.num_to_evict):
            self.assertNotIn(f"Response to '{prompts[i]}'", remaining_responses)

        for i in range(self.num_to_evict, self.watermark_limit + 1):
            self.assertIn(f"Response to '{prompts[i]}'", remaining_responses)

    def test_requests_bypass_cache_during_eviction(self):
        """
        Verify that requests arriving during an active eviction bypass the
        core cache logic and are treated as misses.
        """
        # Spy on the process_request method to see if it's called, while still executing the original method.
        process_request_spy = patch.object(
            self.vcache.vcache_policy,
            "process_request",
            wraps=self.vcache.vcache_policy.process_request,
        ).start()

        # Make the eviction process slow
        original_evict_victims = (
            self.vcache.vcache_config.eviction_policy._evict_victims
        )

        def slow_evict_victims(*args, **kwargs):
            time.sleep(1)
            return original_evict_victims(*args, **kwargs)

        with patch.object(
            self.vcache.vcache_config.eviction_policy,
            "_evict_victims",
            side_effect=slow_evict_victims,
        ):
            # Fill the cache up to just before the watermark limit
            for i in range(self.watermark_limit - 1):
                self.vcache.infer(prompt=f"prompt {i}")

            # Reset the spy to ignore the calls from filling the cache
            process_request_spy.reset_mock()

            # This call will trigger the slow eviction in the background
            self.vcache.infer(prompt="prompt trigger")
            # The spy should be called for the trigger
            self.assertEqual(process_request_spy.call_count, 1)

            # This call should happen during the eviction
            self.vcache.infer(prompt="prompt during eviction")

        # The request during eviction should bypass the core logic,
        # so the call count should still be 1.
        self.assertEqual(process_request_spy.call_count, 1)

    def test_invalid_watermark_resets_to_default(self):
        """
        Verify that creating a policy with an invalid watermark (e.g., > 1)
        causes it to reset to a safe default value.
        """
        with self.assertLogs(
            "vcache.vcache_core.cache.eviction_policy.eviction_policy", level="WARNING"
        ) as cm:
            policy = LRUEvictionPolicy(max_size=10, watermark=1.5)
            self.assertEqual(policy.watermark, 0.95)
            self.assertIn("Watermark must be in (0,1]. Setting to 0.95.", cm.output[0])

    def test_invalid_eviction_percentage_resets_to_default(self):
        """
        Verify that creating a policy with an invalid eviction percentage
        (e.g., < 0) causes it to reset to a safe default value.
        """
        with self.assertLogs(
            "vcache.vcache_core.cache.eviction_policy.eviction_policy", level="WARNING"
        ) as cm:
            policy = LRUEvictionPolicy(max_size=10, eviction_percentage=-0.5)
            self.assertEqual(policy.eviction_percentage, 0.1)
            self.assertIn(
                "Eviction percentage must be in (0,1]. Setting to 0.1.", cm.output[0]
            )

    def test_lock_is_released_on_victim_selection_failure(self):
        """
        Ensure that if an exception occurs during victim selection, the
        eviction lock is released to prevent a deadlock.
        """
        policy = self.vcache.vcache_config.eviction_policy
        with patch.object(
            policy, "select_victims", side_effect=ValueError("Test Exception")
        ):
            with self.assertRaises(ValueError):
                policy.evict(self.vcache.vcache_policy.cache)

            self.assertFalse(policy.is_evicting())

    def test_cache_miss_when_item_evicted_after_knn(self):
        """
        Test the race condition where an item is found by get_knn but is
        evicted before its metadata can be fetched, ensuring a graceful fallback.
        """
        self.patcher.stop()  # Use the real get_knn for this test

        prompt = "test prompt"
        cached_response = "cached response"
        new_response = "new response"

        # Add an item to the cache
        self.vcache.vcache_policy.cache.add(prompt, cached_response, id_set=1)
        knn_result = self.vcache.vcache_policy.cache.get_knn(prompt, k=1)
        self.assertIsNotNone(knn_result)
        embedding_id = knn_result[0][1]

        # Now, evict the item *after* it has been selected by get_knn
        self.vcache.vcache_policy.cache.remove(embedding_id)

        # The policy should now handle the missing metadata gracefully
        with patch.object(
            self.vcache.vcache_policy.inference_engine,
            "create",
            return_value=new_response,
        ) as mock_create:
            is_hit, response, _ = self.vcache.vcache_policy.process_request(
                prompt, None, id_set=1
            )

            self.assertFalse(is_hit)
            self.assertEqual(response, new_response)
            mock_create.assert_called_once()

        self.patcher.start()  # Re-enable the patch for other tests

    def test_shutdown_is_idempotent(self):
        """
        Verify that the shutdown method can be called multiple times without error.
        """
        policy = self.vcache.vcache_config.eviction_policy
        self.assertFalse(policy._is_shutdown)
        policy.shutdown()
        self.assertTrue(policy._is_shutdown)
        policy.shutdown()  # Second call should not raise an error
        self.assertTrue(policy._is_shutdown)

    @patch("atexit.register")
    def test_shutdown_is_registered_on_init(self, mock_atexit_register):
        """
        Verify that the policy's shutdown method is registered with atexit
        during initialization to ensure cleanup.
        """
        policy = LRUEvictionPolicy(max_size=10)
        mock_atexit_register.assert_called_once_with(policy.shutdown)


if __name__ == "__main__":
    unittest.main()
