from typing import List
from vectorq.vectorq_core.cache.eviction_policy.strategy import EvictionPolicyStrategy

class Custom(EvictionPolicyStrategy):
    
    def rank(self, embeddings: List[str]) -> List[int]:
        # TODO
        return []
