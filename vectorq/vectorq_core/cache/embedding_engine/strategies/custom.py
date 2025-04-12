from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.embedding_engine.strategy import EmbeddingEngineStrategy

class Custom(EmbeddingEngineStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
    
    def get_embedding(self, text: str) -> List[float]:
        # TODO
        return []
