from typing import Dict, List, override

from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class BenchmarkEmbeddingEngine(EmbeddingEngine):
    """
    An embedding engine implementation that returns pre-computed embeddings for given texts.
    It is used for benchmarking purposes.
    """

    def __init__(self, text_embedding_map: Dict[str, List[float]]):
        """
        Initialize the benchmark embedding engine with predefined embeddings.

        Args:
            text_embedding_map: A dictionary mapping text strings to their pre-computed embeddings.
                               Keys are text strings and values are embedding vectors.
        """
        super().__init__()
        self.text_embedding_map = text_embedding_map

    @override
    def get_embedding(self, text: str) -> List[float]:
        if text not in self.text_embedding_map:
            raise ValueError(f"Text '{text}' not found in text_embedding_map")
        return self.text_embedding_map[text]
