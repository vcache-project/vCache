from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.embedding_engine.strategy import EmbeddingEngineStrategy
from vectorq.vectorq_core.cache.embedding_engine.strategies.lang_chain import LangChain
from vectorq.vectorq_core.cache.embedding_engine.strategies.open_ai import OpenAI

class EmbeddingEngineType(Enum):
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    
class EmbeddingEngine():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        
        self.strategy: EmbeddingEngineStrategy = None
        
        match self.vectorq_config._embedding_engine_type:
            case EmbeddingEngineType.LANGCHAIN:
                self.strategy: EmbeddingEngineStrategy = LangChain(vectorq_config=self.vectorq_config)
            case EmbeddingEngineType.OPENAI:
                self.strategy: EmbeddingEngineStrategy = OpenAI(vectorq_config=self.vectorq_config)
            case _:
                raise ValueError(f"Invalid embedding engine type")
            
    def get_embedding(self, text: str) -> List[float]:
        return self.strategy.get_embedding(text)
