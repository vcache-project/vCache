import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from vcache.vcache_core.cache.eviction_policy.strategies.fifo import (
    FIFOEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.lru import (
    LRUEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.mru import (
    MRUEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.scu import (
    SCUEvictionPolicy,
)


class TestEvictionPolicyStrategies(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing eviction policies."""
        self.max_size = 10
        self.eviction_percentage = 0.2
        self.num_to_evict = int(self.max_size * self.eviction_percentage)

        self.metadata = []
        for i in range(5):
            mock_meta = MagicMock()
            mock_meta.embedding_id = i
            mock_meta.created_at = datetime.now(timezone.utc)
            mock_meta.last_accessed = mock_meta.created_at
            self.metadata.append(mock_meta)
            time.sleep(0.01)  # Ensure timestamps are distinct

    def test_fifo_eviction(self):
        """Test the FIFO eviction strategy."""
        policy = FIFOEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)
        victims = policy.select_victims(self.metadata)

        # FIFO should evict the first items added (0 and 1)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [0, 1])

    def test_lru_eviction(self):
        """Test the LRU eviction strategy."""
        policy = LRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Simulate access to items 0 and 1, making them the most recently used
        policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[1])

        victims = policy.select_victims(self.metadata)

        # LRU should evict the least recently used items (2 and 3)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])

    def test_mru_eviction(self):
        """Test the MRU eviction strategy."""
        policy = MRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Simulate access to items 3 and 4, making them the most recently used
        policy.update_eviction_metadata(self.metadata[3])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[4])

        victims = policy.select_victims(self.metadata)

        # MRU should evict the most recently used items (3 and 4)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [3, 4])

    def test_no_eviction(self):
        """Test the NoEviction policy."""
        policy = NoEvictionPolicy()
        victims = policy.select_victims(self.metadata)

        # NoEviction policy should never select any victims
        self.assertEqual(len(victims), 0)

    def test_scu_eviction(self):
        """Test the SCU eviction strategy."""
        policy = SCUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Manually set t_prime and observations for a controlled test
        # Item 0: Proven Loser (high t_prime, high n_obs) -> High distance
        self.metadata[0].t_prime = 0.9
        self.metadata[0].observations = [0] * 10

        # Item 1: Suspected Loser (high t_prime, low n_obs) -> Mid distance
        self.metadata[1].t_prime = 0.9
        self.metadata[1].observations = [0] * 2

        # Item 2: Proven Winner (low t_prime, high n_obs) -> Low distance
        self.metadata[2].t_prime = 0.1
        self.metadata[2].observations = [0] * 10

        # Item 3: Suspected Winner (low t_prime, low n_obs) -> Mid-low distance
        self.metadata[3].t_prime = 0.1
        self.metadata[3].observations = [0] * 2

        # Item 4: No t_prime, should be considered a "suspected loser"
        self.metadata[4].t_prime = None

        victims = policy.select_victims(self.metadata)

        # Expected victims:
        # 1. Item 4 (infinite distance because t_prime is None)
        # 2. Item 1 (Suspected Loser, has a larger distance than the Proven Loser)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [1, 4])

    def test_scu_fallback_eviction(self):
        """Test the SCU fallback to LRU when no t_prime is available."""
        scu_policy = SCUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)
        lru_policy = LRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Ensure all t_prime values are None
        for meta in self.metadata:
            meta.t_prime = None

        # Update last_accessed to create a clear LRU order
        # Items 2 and 3 will be the least recently used
        lru_policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        lru_policy.update_eviction_metadata(self.metadata[1])
        time.sleep(0.01)
        lru_policy.update_eviction_metadata(self.metadata[4])

        victims = scu_policy.select_victims(self.metadata)

        # Fallback should use LRU, evicting the least recently used items
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])


if __name__ == "__main__":
    unittest.main()
