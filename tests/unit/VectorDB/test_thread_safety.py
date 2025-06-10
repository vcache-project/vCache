import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.chroma import (
    ChromaVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.faiss import (
    FAISSVectorDB,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
)


class TestVectorDBThreadSafety(unittest.TestCase):
    """Test thread safety of vector database implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.num_threads = 10
        self.embeddings_per_thread = 20
        
    def _generate_random_embedding(self, seed: int) -> List[float]:
        """Generate a deterministic random embedding based on seed."""
        import random
        random.seed(seed)
        return [random.random() for _ in range(self.embedding_dim)]

    def _add_embeddings_worker(self, vector_db, start_seed: int, count: int) -> Set[int]:
        """Worker function to add embeddings in a thread."""
        added_ids = set()
        for i in range(count):
            embedding = self._generate_random_embedding(start_seed + i)
            embedding_id = vector_db.add(embedding)
            added_ids.add(embedding_id)
        return added_ids

    def test_hnswlib_thread_safety(self):
        """Test HNSWLibVectorDB thread safety."""
        vector_db = HNSWLibVectorDB(similarity_metric_type=SimilarityMetricType.COSINE)
        self._test_vector_db_thread_safety(vector_db)

    def test_chroma_thread_safety(self):
        """Test ChromaVectorDB thread safety."""
        vector_db = ChromaVectorDB(similarity_metric_type=SimilarityMetricType.COSINE)
        self._test_vector_db_thread_safety(vector_db)

    def test_faiss_thread_safety(self):
        """Test FAISSVectorDB thread safety."""
        vector_db = FAISSVectorDB(similarity_metric_type=SimilarityMetricType.COSINE)
        self._test_vector_db_thread_safety(vector_db)

    def _test_vector_db_thread_safety(self, vector_db):
        """Generic test for vector database thread safety."""
        all_ids = set()
        
        # Use ThreadPoolExecutor to simulate concurrent access
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit tasks to add embeddings concurrently
            futures = []
            for i in range(self.num_threads):
                start_seed = i * self.embeddings_per_thread
                future = executor.submit(
                    self._add_embeddings_worker, 
                    vector_db, 
                    start_seed, 
                    self.embeddings_per_thread
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                thread_ids = future.result()
                # Check for ID collisions
                intersection = all_ids.intersection(thread_ids)
                self.assertEqual(
                    len(intersection), 0, 
                    f"ID collision detected: {intersection}"
                )
                all_ids.update(thread_ids)
        
        # Verify total number of unique IDs
        expected_total = self.num_threads * self.embeddings_per_thread
        self.assertEqual(
            len(all_ids), expected_total,
            f"Expected {expected_total} unique IDs, got {len(all_ids)}"
        )
        
        # Verify IDs are sequential (0 to expected_total - 1)
        expected_ids = set(range(expected_total))
        self.assertEqual(
            all_ids, expected_ids,
            "IDs are not sequential or have gaps"
        )
        
        # Test that we can query the database
        test_embedding = self._generate_random_embedding(999)
        results = vector_db.get_knn(test_embedding, k=5)
        self.assertLessEqual(len(results), 5, "Should return at most 5 results")
        
        # Verify that all returned IDs exist in our added IDs
        for similarity, embedding_id in results:
            self.assertIn(
                embedding_id, all_ids,
                f"Returned ID {embedding_id} was not in added IDs"
            )

    def test_concurrent_add_and_query(self):
        """Test concurrent add and query operations."""
        vector_db = HNSWLibVectorDB(similarity_metric_type=SimilarityMetricType.COSINE)
        
        def add_worker():
            """Worker that adds embeddings."""
            for i in range(10):
                embedding = self._generate_random_embedding(i)
                vector_db.add(embedding)
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        def query_worker():
            """Worker that queries embeddings."""
            query_embedding = self._generate_random_embedding(999)
            for _ in range(10):
                try:
                    results = vector_db.get_knn(query_embedding, k=3)
                    # Should not raise exceptions
                    self.assertIsInstance(results, list)
                except Exception as e:
                    self.fail(f"Query worker failed with exception: {e}")
                time.sleep(0.001)
        
        # Run concurrent add and query operations
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=add_worker))
            threads.append(threading.Thread(target=query_worker))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final state
        self.assertFalse(vector_db.is_empty())

    def test_concurrent_reset_operations(self):
        """Test concurrent reset operations don't cause issues."""
        vector_db = HNSWLibVectorDB(similarity_metric_type=SimilarityMetricType.COSINE)
        
        # Add some initial data
        for i in range(5):
            embedding = self._generate_random_embedding(i)
            vector_db.add(embedding)
        
        def reset_worker():
            """Worker that resets the database."""
            vector_db.reset()
        
        def add_worker():
            """Worker that adds embeddings."""
            for i in range(5):
                embedding = self._generate_random_embedding(100 + i)
                try:
                    vector_db.add(embedding)
                except Exception:
                    # Reset might have happened, which is fine
                    pass
        
        # Run concurrent operations
        threads = []
        threads.append(threading.Thread(target=reset_worker))
        threads.append(threading.Thread(target=add_worker))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not crash and database should be in a valid state
        self.assertIsInstance(vector_db.is_empty(), bool)


if __name__ == "__main__":
    unittest.main() 