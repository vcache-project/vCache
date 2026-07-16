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

    def test_persists_files_to_disk(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        vector_db.add(embedding=[0.1, 0.2, 0.3])

        assert os.path.exists(self.persist_path)
        assert os.path.exists(self.persist_path + ".meta.json")

    def test_survives_simulated_restart(self):
        """Embeddings added by one instance should be visible to a fresh
        instance pointed at the same path, simulating a process restart."""
        first_instance = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        first_instance.add(embedding=[0.1, 0.2, 0.3])
        first_instance.add(embedding=[0.2, 0.3, 0.4])
        del first_instance

        second_instance = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        knn = second_instance.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 2

        # New adds after reload should not collide with restored ids
        new_id = second_instance.add(embedding=[0.3, 0.4, 0.5])
        assert new_id == 2

    def test_reset(self):
        vector_db = PersistentHNSWLibVectorDB(persist_path=self.persist_path)
        vector_db.add(embedding=[0.1, 0.2, 0.3])
        vector_db.add(embedding=[0.2, 0.3, 0.4])

        vector_db.reset()

        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=3)
        assert len(knn) == 0

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
