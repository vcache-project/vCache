from typing import List, TYPE_CHECKING
from langchain_community.embeddings import HuggingFaceEmbeddings

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.cache.embedding_engine.strategy import EmbeddingEngineStrategy

class LangChain(EmbeddingEngineStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.vectorq_config._embedding_engine_model_name
        )
        
    def get_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from LangChain: {e}")
