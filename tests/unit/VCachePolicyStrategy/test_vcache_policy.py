import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from vcache.config import VCacheConfig
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_policy.strategies.verified import (
    VerifiedDecisionPolicy,
    _Action,
)


class TestVerifiedDecisionPolicy(unittest.TestCase):
    def setUp(self):
        """Set up a new policy and mock dependencies for each test."""
        self.mock_inference_engine = MagicMock()
        self.mock_similarity_evaluator = MagicMock()
        self.mock_cache = MagicMock()

        # Create a stateful mock for the cache
        self.mock_metadata_store = {}
        self.next_embedding_id = 0

        def add_to_cache(prompt, response, id_set=None):
            self.next_embedding_id += 1
            # Simulate adding metadata
            mock_meta = MagicMock(spec=EmbeddingMetadataObj)
            mock_meta.response = response
            mock_meta.observations = []
            mock_meta.id_set = id_set
            self.mock_metadata_store[self.next_embedding_id] = mock_meta
            return self.next_embedding_id

        def get_metadata(embedding_id):
            # Allow raising KeyError to simulate not found
            if embedding_id not in self.mock_metadata_store:
                raise KeyError("Metadata not found")
            return self.mock_metadata_store[embedding_id]

        def update_metadata(embedding_id, embedding_metadata):
            self.mock_metadata_store[embedding_id] = embedding_metadata

        self.mock_cache.get_metadata.side_effect = get_metadata
        self.mock_cache.update_metadata.side_effect = update_metadata
        self.mock_cache.add.side_effect = add_to_cache

        mock_config = MagicMock(spec=VCacheConfig)
        mock_config.inference_engine = self.mock_inference_engine
        mock_config.similarity_evaluator = self.mock_similarity_evaluator
        # Add all required attributes for Cache creation
        mock_config.embedding_engine = MagicMock()
        mock_config.embedding_metadata_storage = MagicMock()
        mock_config.vector_db = MagicMock()
        mock_config.eviction_policy = MagicMock()

        self.policy = VerifiedDecisionPolicy()
        self.policy.setup(mock_config)

        # After setup, replace the real cache with our stateful mock
        self.policy.cache = self.mock_cache

    def tearDown(self):
        """Shutdown the policy to clean up threads."""
        self.policy.shutdown()

    def test_empty_cache_is_miss(self):
        """Test that the first request to an empty cache is a miss."""
        self.mock_cache.get_knn.return_value = []
        self.mock_inference_engine.create.return_value = "new response"

        is_hit, response, _ = self.policy.process_request("prompt", None, id_set=1)

        self.assertFalse(is_hit)
        self.assertEqual(response, "new response")
        self.mock_cache.add.assert_called_once_with(
            prompt="prompt", response="new response", id_set=1
        )

    @patch("vcache.vcache_policy.strategies.verified._Algorithm.select_action")
    def test_exploit_is_cache_hit(self, mock_select_action):
        """Test that an EXPLOIT action results in a cache hit."""
        mock_select_action.return_value = _Action.EXPLOIT
        self.mock_cache.get_knn.return_value = [(0.95, 1)]
        # Pre-populate cache
        mock_meta = MagicMock(spec=EmbeddingMetadataObj)
        mock_meta.response = "cached response"
        mock_meta.observations = []
        mock_meta.id_set = 1
        self.mock_metadata_store[1] = mock_meta

        is_hit, response, _ = self.policy.process_request("prompt", None, id_set=1)

        self.assertTrue(is_hit)
        self.assertEqual(response, "cached response")
        self.mock_inference_engine.create.assert_not_called()

    @patch("vcache.vcache_policy.strategies.verified._Algorithm.select_action")
    def test_explore_updates_in_background(self, mock_select_action):
        """Test that an EXPLORE action queues a background update."""
        mock_select_action.return_value = _Action.EXPLORE
        self.mock_cache.get_knn.return_value = [(0.8, 1)]

        mock_meta = MagicMock(spec=EmbeddingMetadataObj)
        mock_meta.response = "cached response"
        mock_meta.observations = []
        mock_meta.id_set = 1
        self.mock_metadata_store[1] = mock_meta

        self.mock_inference_engine.create.return_value = "new response"
        self.mock_similarity_evaluator.answers_similar.return_value = True

        is_hit, response, _ = self.policy.process_request("prompt", None, id_set=1)
        self.policy.shutdown()  # Wait for background tasks to finish

        self.assertFalse(is_hit)
        self.assertEqual(response, "new response")
        # Check that the background update was performed
        self.assertEqual(len(mock_meta.observations), 1)
        self.assertEqual(mock_meta.observations[0], (0.8, 1))

    def test_concurrent_add_and_read_stability(self):
        """
        Stress-test the system's stability with concurrent reads and writes.
        This test validates the atomic add fix (C1) and the graceful read
        failure fix (C2).
        """
        num_threads = 10
        ops_per_thread = 20

        def writer_task(i):
            for j in range(ops_per_thread):
                prompt = f"writer-{i}-prompt-{j}"
                self.mock_cache.get_knn.return_value = []  # Force miss
                self.mock_inference_engine.create.return_value = "new response"
                self.policy.process_request(prompt, None, id_set=1)
            return True

        def reader_task(i):
            for _ in range(ops_per_thread):
                self.mock_cache.get_knn.return_value = [(0.9, 1)]  # Force hit
                self.policy.process_request("reader-prompt", None, id_set=1)
            return True

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            # Half writers, half readers
            for i in range(num_threads // 2):
                futures.append(executor.submit(writer_task, i))
                futures.append(executor.submit(reader_task, i))

            for future in as_completed(futures):
                # The test passes if no exceptions were raised from the threads.
                self.assertTrue(future.result())

    def test_concurrent_explore_and_update_integrity(self):
        """
        Stress-test the background update logic for data integrity (C4) and
        resilience against eviction (C3).
        """
        num_threads = 20
        ops_per_thread = 5

        # Pre-populate cache
        mock_meta = MagicMock(spec=EmbeddingMetadataObj)
        mock_meta.response = "cached"
        mock_meta.observations = []
        mock_meta.id_set = 1
        self.mock_metadata_store[1] = mock_meta

        with patch.object(
            self.policy.bayesian, "select_action", return_value=_Action.EXPLORE
        ):

            def explore_task():
                self.mock_cache.get_knn.return_value = [(0.8, 1)]
                self.mock_inference_engine.create.return_value = "new response"
                self.mock_similarity_evaluator.answers_similar.return_value = True
                self.policy.process_request("similar_prompt", None, id_set=1)
                return True

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(explore_task)
                    for _ in range(num_threads * ops_per_thread)
                ]
                for future in as_completed(futures):
                    self.assertTrue(future.result())

        # We must shut down to ensure all queued updates are processed
        self.policy.shutdown()

        # The final number of observations should be exactly the total number of tasks
        expected_observations = num_threads * ops_per_thread
        self.assertEqual(len(mock_meta.observations), expected_observations)

    def test_race_condition_read_side_graceful_failure(self):
        """Test graceful failure when metadata disappears after knn."""
        self.mock_cache.get_knn.return_value = [(0.9, 1)]
        # Simulate metadata not being found by raising an error
        self.mock_cache.get_metadata.side_effect = KeyError("Metadata not found")
        self.mock_inference_engine.create.return_value = "fallback response"

        is_hit, response, _ = self.policy.process_request("prompt", None, id_set=1)

        self.assertFalse(is_hit)
        self.assertEqual(response, "fallback response")
        # It should have been treated as a cache miss, so add is called
        self.mock_cache.add.assert_called_once_with(
            prompt="prompt", response="fallback response", id_set=1
        )

    def test_race_condition_update_vs_eviction(self):
        """Test graceful failure when metadata is evicted before background update."""
        with patch.object(
            self.policy.bayesian, "select_action", return_value=_Action.EXPLORE
        ):
            self.mock_cache.get_knn.return_value = [(0.8, 1)]
            mock_meta = MagicMock(spec=EmbeddingMetadataObj)
            mock_meta.response = "cached"
            mock_meta.observations = []
            mock_meta.id_set = 1
            self.mock_metadata_store[1] = mock_meta
            self.mock_inference_engine.create.return_value = "new response"

            # In the background task, simulate metadata having been evicted
            original_get_metadata = self.mock_cache.get_metadata.side_effect

            def get_metadata_for_update(embedding_id):
                # First call from main thread works
                if self.mock_cache.get_metadata.call_count == 1:
                    return original_get_metadata(embedding_id)
                # Second call from background worker fails
                return None

            self.mock_cache.get_metadata.side_effect = get_metadata_for_update

            # This call will queue the background update
            self.policy.process_request("prompt", None, id_set=1)

            # Allow background tasks to run and complete
            self.policy.shutdown()

            # The test passes if no exception was raised during the background update
            # and the original observations list is unchanged
            self.assertEqual(len(mock_meta.observations), 0)


if __name__ == "__main__":
    unittest.main()
