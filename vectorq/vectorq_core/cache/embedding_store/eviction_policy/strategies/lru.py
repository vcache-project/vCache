from typing import List
from vectorq.vectorq_core.cache.embedding_store.eviction_policy.strategy import EvictionPolicyStrategy
from collections import OrderedDict

class LRU():

    def __init__(self, max_elements: int):
        print("LRU Eviction Policy")
        self.max_elements = max_elements
        self.embedding_count = 0
        self.next_embedding_id = 0

        # OrderedMap<Embeddings_Int_Index, Cluster>
        self.map_embedding_id_to_freq = OrderedDict()

    def call_eviction_policy(self):
        if self.embedding_count < self.max_elements:
            self.embedding_count += 1
            self.next_embedding_id += 1
            self.map_embedding_id_to_freq[self.next_embedding_id - 1] = 1
            return False, self.next_embedding_id - 1
        else:
            evicted_id = next(iter(self.map_embedding_id_to_freq))
            self.map_embedding_id_to_freq.move_to_end(evicted_id)
            # self.embedding_count -= 1 #### check if you incremente later
            return True, evicted_id
    
    def promote(self, embedding_id: int):
        self.map_embedding_id_to_freq.move_to_end(embedding_id)

    def reset(self):
        self.map_embedding_id_to_freq.clear()
        self.embedding_count = 0
        self.next_embedding_id = 0
    
    def is_empty(self) -> bool:
        return self.embedding_count == 0
    

