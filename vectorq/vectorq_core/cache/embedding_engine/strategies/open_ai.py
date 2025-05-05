from typing import List, Optional

from openai import OpenAI as OpenAIClient

from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class OpenAIEmbeddingEngine(EmbeddingEngine):
    """
    OpenAI implementation of embedding engine
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        """
        Initialize an OpenAI embedding engine

        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: Optional API key (if not provided, will use environment variables)
        """
        self.model_name = model_name
        self.client = OpenAIClient(api_key=api_key) if api_key else OpenAIClient()

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the provided text using OpenAI's API

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from OpenAI: {e}")
