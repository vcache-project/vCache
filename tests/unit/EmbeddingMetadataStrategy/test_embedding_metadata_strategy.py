import unittest
from concurrent.futures import ThreadPoolExecutor

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class TestEmbeddingMetadataStorageStrategy(unittest.TestCase):
    def test_in_memory_strategy(self):
        embedding_metadata_storage = InMemoryEmbeddingMetadataStorage()

        initial_obj = EmbeddingMetadataObj(embedding_id=0, response="test")
        embedding_id = embedding_metadata_storage.add_metadata(
            embedding_id=0, metadata=initial_obj
        )
        assert embedding_id == 0
        assert embedding_metadata_storage.get_metadata(embedding_id=0) == initial_obj

        updated_obj = EmbeddingMetadataObj(embedding_id=0, response="test2")
        embedding_metadata_storage.update_metadata(embedding_id=0, metadata=updated_obj)
        assert embedding_metadata_storage.get_metadata(embedding_id=0) == updated_obj

        embedding_metadata_storage.flush()
        with self.assertRaises(ValueError):
            embedding_metadata_storage.get_metadata(embedding_id=0)


class TestInMemoryEmbeddingMetadataStorageThreadSafety(unittest.TestCase):
    def test_concurrent_add_observation(self):
        storage = InMemoryEmbeddingMetadataStorage()
        embedding_id = 0
        initial_obj = EmbeddingMetadataObj(embedding_id=embedding_id, response="test")
        storage.add_metadata(embedding_id=embedding_id, metadata=initial_obj)

        num_threads = 10
        num_observations_per_thread = 100
        total_observations = num_threads * num_observations_per_thread

        def add_observations_task(storage, embedding_id):
            for i in range(num_observations_per_thread):
                observation = (float(i), 1)
                storage.add_observation(embedding_id, observation)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(add_observations_task, storage, embedding_id)
                for _ in range(num_threads)
            ]
            for future in futures:
                future.result()

        metadata = storage.get_metadata(embedding_id)
        # The initial object has 2 observations, so we add the total observations to that.
        self.assertEqual(len(metadata.observations), total_observations + 2)


if __name__ == "__main__":
    unittest.main()
