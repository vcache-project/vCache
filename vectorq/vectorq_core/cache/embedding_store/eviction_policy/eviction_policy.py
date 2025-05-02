from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.embedding_store.eviction_policy.strategy import EvictionPolicyStrategy
from vectorq.vectorq_core.cache.embedding_store.eviction_policy.strategies.lru import LRU
from vectorq.vectorq_core.cache.embedding_store.eviction_policy.strategies.lfu import LFU

class EvictionPolicyType(Enum):
    NONE = "none"
    LRU = "lru"
    LFU = "lfu"

class EvictionPolicy():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.execution_interval: datetime = None
        self._eviction_policy_type: EvictionPolicyType = self.vectorq_config.eviction_policy
        self.max_elements = self.vectorq_config.capacity
        
        match self._eviction_policy_type:
            case EvictionPolicyType.NONE:
                self.strategy: EvictionPolicyStrategy = None
            case EvictionPolicyType.LRU:
                self.strategy: EvictionPolicyStrategy = LRU(self.max_elements)
            case EvictionPolicyType.LFU:
                self.strategy: EvictionPolicyStrategy = LFU(self.max_elements)
            case _:
                raise ValueError(f"Invalid eviction policy type")
    
    def call_eviction_policy(self):
        if self.strategy is None:
            return False, -1
        else:
            return self.strategy.call_eviction_policy()
    
    def promote(self, embedding_id: int):
        if self.strategy is not None:
            self.strategy.promote(embedding_id)
        
    
    def is_empty(self) -> bool:
        if self.strategy is not None:
            return self.strategy.is_empty()
        else:
            return True
    
    def reset(self):
        if self.strategy is not None:
            self.strategy.reset()