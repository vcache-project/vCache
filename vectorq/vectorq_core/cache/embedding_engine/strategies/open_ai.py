from openai import OpenAI as OpenAIClient
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.embedding_engine.strategy import EmbeddingEngineStrategy

class OpenAI(EmbeddingEngineStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
        self.client = OpenAIClient()
        
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input = text,
                model = self.vectorq_config._embedding_engine_model_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from OpenAI: {e}")
