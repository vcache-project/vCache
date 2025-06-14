import unittest

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


if __name__ == "__main__":
    unittest.main()
