import os
import tempfile
import unittest

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    SQLiteEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)


class TestSQLiteEmbeddingMetadataStorage(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: on Windows, sqlite3 keeps the file handle
        # open for the lifetime of the connection, which otherwise makes
        # tearDown's rmtree fail with a PermissionError.
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.tmp_dir.name, "metadata.sqlite3")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_sqlite_strategy(self):
        embedding_metadata_storage = SQLiteEmbeddingMetadataStorage(self.db_path)

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

    def test_get_missing_metadata_raises(self):
        embedding_metadata_storage = SQLiteEmbeddingMetadataStorage(self.db_path)
        with self.assertRaises(ValueError):
            embedding_metadata_storage.get_metadata(embedding_id=42)

    def test_update_missing_metadata_raises(self):
        embedding_metadata_storage = SQLiteEmbeddingMetadataStorage(self.db_path)
        with self.assertRaises(ValueError):
            embedding_metadata_storage.update_metadata(
                embedding_id=42,
                metadata=EmbeddingMetadataObj(embedding_id=42, response="test"),
            )

    def test_remove_metadata(self):
        embedding_metadata_storage = SQLiteEmbeddingMetadataStorage(self.db_path)
        embedding_metadata_storage.add_metadata(
            embedding_id=1,
            metadata=EmbeddingMetadataObj(embedding_id=1, response="test"),
        )
        assert embedding_metadata_storage.remove_metadata(embedding_id=1) is True
        assert embedding_metadata_storage.remove_metadata(embedding_id=1) is False

    def test_get_all_embedding_metadata_objects(self):
        embedding_metadata_storage = SQLiteEmbeddingMetadataStorage(self.db_path)
        for i in range(3):
            embedding_metadata_storage.add_metadata(
                embedding_id=i,
                metadata=EmbeddingMetadataObj(embedding_id=i, response=f"test{i}"),
            )
        all_metadata = embedding_metadata_storage.get_all_embedding_metadata_objects()
        assert len(all_metadata) == 3
        assert {meta.response for meta in all_metadata} == {"test0", "test1", "test2"}

    def test_survives_simulated_restart(self):
        """Data written by one instance should be visible to a fresh instance
        pointed at the same file, simulating a process restart."""
        first_instance = SQLiteEmbeddingMetadataStorage(self.db_path)
        first_instance.add_metadata(
            embedding_id=7,
            metadata=EmbeddingMetadataObj(
                embedding_id=7, response="persisted", id_set=3
            ),
        )
        del first_instance

        second_instance = SQLiteEmbeddingMetadataStorage(self.db_path)
        restored = second_instance.get_metadata(embedding_id=7)
        assert restored.response == "persisted"
        assert restored.id_set == 3


if __name__ == "__main__":
    unittest.main()
