from typing import List
from vectorq.vectorq_core.cache.embedding_store.eviction_policy.strategy import EvictionPolicyStrategy
from collections import OrderedDict

class LFU():

    def __init__(self, max_elements: int):
        self.max_elements = max_elements
        self.embedding_count = 0
        self.next_embedding_id = 0
        
        # OrderedMap<Embeddings_Int_Index, Cluster>
        self.map_embedding_id_to_freq = OrderedDict()
        self.map_frequency_to_embedding_id = {}
        self.min_freq = 0 


    def call_eviction_policy(self):
        if self.embedding_count < self.max_elements:
            self.embedding_count += 1
            self.next_embedding_id += 1
            insert_id = self.next_embedding_id - 1
            self.map_embedding_id_to_freq[insert_id] = 1
            if 1 not in self.map_frequency_to_embedding_id:
                self.map_frequency_to_embedding_id[1] = OrderedDict({insert_id: None})
            else:
                self.map_frequency_to_embedding_id[1][insert_id] = None
            self.min_freq = 1
            return False, insert_id
        else:
            embedding_id, _ = self.map_frequency_to_embedding_id[self.min_freq].popitem(last=False)
            self.map_embedding_id_to_freq[embedding_id] = 1
            if 1 not in self.map_frequency_to_embedding_id:
                self.map_frequency_to_embedding_id[1] = OrderedDict({embedding_id: None})
            else:
                self.map_frequency_to_embedding_id[1][embedding_id] = None
            self.min_freq = 1
            return True, embedding_id
    
    def promote(self, embedding_id: int):
        embedding_freq = self.map_embedding_id_to_freq[embedding_id]
        self.map_frequency_to_embedding_id[embedding_freq].pop(embedding_id)
        if len(self.map_frequency_to_embedding_id[embedding_freq]) == 0:
            del self.map_frequency_to_embedding_id[embedding_freq]
            if embedding_freq == self.min_freq:
                self.min_freq += 1
        embedding_freq += 1
        self.map_embedding_id_to_freq[embedding_id] = embedding_freq
        if embedding_freq not in self.map_frequency_to_embedding_id:
            self.map_frequency_to_embedding_id[embedding_freq] = OrderedDict({embedding_id: None})
        else:
            self.map_frequency_to_embedding_id[embedding_freq][embedding_id] = None


    def reset(self):
        self.map_embedding_id_to_freq.clear()
        self.embedding_count = 0
        self.next_embedding_id = 0
        self.map_frequency_to_embedding_id = {}
        self.min_freq = 0 
    
    def is_empty(self) -> bool:
        return self.embedding_count == 0