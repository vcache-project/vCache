from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.eviction_policy.strategy import EvictionPolicyStrategy
from vectorq.vectorq_core.cache.eviction_policy.strategies.lru import LRU

class EvictionPolicyType(Enum):
    NONE = "none"
    LRU = "lru"

class EvictionPolicy():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.execution_interval: datetime = None
        self._eviction_policy_type: EvictionPolicyType = self.vectorq_config._eviction_policy_type
        
        match self._eviction_policy_type:
            case EvictionPolicyType.NONE:
                self.strategy: EvictionPolicyStrategy = None
            case EvictionPolicyType.LRU:
                self.strategy: EvictionPolicyStrategy = LRU()
            case _:
                raise ValueError(f"Invalid eviction policy type")
    
    def evict(self) -> List[str]:
        # TODO
        pass
    
    def rank(self, embeddings: List[str]) -> List[int]:
        # TODO
        pass
