import json
import os
import tempfile
import unittest

from vcache.vcache_core.cache.embedding_store.vector_db import (
    PersistentHNSWLibVectorDB,
    SimilarityMetricType,
)


class TestPersistentHNSWLibVectorDB(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.persist_path = os.path.join(self.tmp_dir.name, "index.hnsw")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_add_and_get_knn(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)

        embedding = [0.1, 0.2, 0.3]
        id1 = vector_db.add(embedding=embedding)
        knn = vector_db.get_knn(embedding=embedding, k=1)
        assert len(knn) == 1
        assert knn[0][1] == id1

        vector_db.add(embedding=[0.2, 0.3, 0.4])
        vector_db.add(embedding=[0.3, 0.4, 0.5])
        knn = vector_db.get_knn(embedding=embedding, k=3)
        assert len(knn) == 3

    def test_remove(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)

        id1 = vector_db.add(embedding=[0.1, 0.2, 0.3])
        id2 = vector_db.add(embedding=[0.2, 0.3, 0.4])
        vector_db.remove(embedding_id=id1)

        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 1
        assert knn[0][1] == id2

    def test_mutations_below_checkpoint_interval_only_touch_wal(self):
        """Below the checkpoint interval, mutations should be captured by
        cheap WAL appends rather than a full index rewrite, so no manifest
        or index snapshot should exist yet."""
        vector_db = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        vector_db.add(embedding=[0.1, 0.2, 0.3])

        assert os.path.exists(self.persist_path + ".wal")
        assert os.path.getsize(self.persist_path + ".wal") > 0
        assert not os.path.exists(self.persist_path + ".manifest.json")

    def test_checkpoint_publishes_manifest_and_clears_wal(self):
        vector_db = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.checkpoint()

        assert os.path.exists(self.persist_path + ".manifest.json")
        assert os.path.exists(self.persist_path + ".g1")
        assert os.path.getsize(self.persist_path + ".wal") == 0
        # No leftover temp files from the atomic rename.
        assert not os.path.exists(self.persist_path + ".g1.tmp")
        assert not os.path.exists(self.persist_path + ".manifest.json.tmp")

    def test_auto_checkpoints_after_interval_and_prunes_old_generation(self):
        vector_db = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=2
        )
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.add(embedding=[0.2, 0.3, 0.4])

        # Auto-checkpoint should have fired after the 2nd mutation.
        assert os.path.exists(self.persist_path + ".manifest.json")
        assert os.path.exists(self.persist_path + ".g1")
        assert os.path.getsize(self.persist_path + ".wal") == 0

        vector_db.add(embedding=[0.3, 0.4, 0.5])
        vector_db.add(embedding=[0.4, 0.5, 0.6])

        # A second checkpoint should have fired and the stale generation
        # file from the first checkpoint should have been pruned.
        assert os.path.exists(self.persist_path + ".g2")
        assert not os.path.exists(self.persist_path + ".g1")

    def test_survives_simulated_restart_before_any_checkpoint(self):
        """Mutations that never reached the checkpoint interval should still
        survive a restart, recovered by replaying the WAL."""
        first_instance = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        first_instance.add(embedding=[0.1, 0.2, 0.3])
        first_instance.add(embedding=[0.2, 0.3, 0.4])
        del first_instance

        second_instance = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        knn = second_instance.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 2

        # New adds after reload should not collide with restored ids
        new_id = second_instance.add(embedding=[0.3, 0.4, 0.5])
        assert new_id == 2

    def test_survives_simulated_restart_after_checkpoint(self):
        first_instance = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=1
        )
        first_instance.add(embedding=[0.1, 0.2, 0.3])
        first_instance.add(embedding=[0.2, 0.3, 0.4])
        del first_instance

        second_instance = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=1
        )
        knn = second_instance.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 2

    def test_recovers_from_crash_between_checkpoint_index_write_and_manifest(self):
        """If the process dies after the new generation's index file is
        written but before the manifest is rewritten to point at it, the
        on-disk manifest still refers to the previous, fully consistent
        generation. Recovery must fall back to it (plus WAL replay) rather
        than surface the orphaned, half-committed generation."""
        vector_db = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.checkpoint()  # generation 1 fully committed

        vector_db.add(embedding=[0.2, 0.3, 0.4])  # queued in the WAL

        def failing_write_manifest():
            raise RuntimeError("simulated crash before manifest commit")

        vector_db._write_manifest = failing_write_manifest
        with self.assertRaises(RuntimeError):
            vector_db.checkpoint()

        # The new generation's index snapshot made it to disk...
        assert os.path.exists(self.persist_path + ".g2")
        # ...but the manifest was never updated to point at it.
        with open(vector_db._manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest["generation"] == 1
        del vector_db

        recovered = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path, checkpoint_interval=100
        )
        # Recovers via generation 1 + WAL replay, ignoring the orphaned
        # generation 2 snapshot that was never published.
        knn = recovered.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 2

    def test_reset(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.add(embedding=[0.2, 0.3, 0.4])

        vector_db.reset()

        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=3)
        assert len(knn) == 0

    def test_reset_persists_across_restart(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.reset()
        del vector_db

        reloaded = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        assert reloaded.is_empty()

    def test_euclidean_metric(self):
        vector_db = PersistentHNSWLibVectorDB(
            persist_path=self.persist_path,
            similarity_metric_type=SimilarityMetricType.EUCLIDEAN,
        )
        embedding = [0.1, 0.2, 0.3]
        id1 = vector_db.add(embedding=embedding)
        knn = vector_db.get_knn(embedding=embedding, k=1)
        assert knn[0][1] == id1


if __name__ == "__main__":
    unittest.main()
