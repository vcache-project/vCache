from typing import List, override

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class BenchmarkEmbeddingEngine(EmbeddingEngine):
    """
    An embedding engine implementation that returns pre-computed embeddings for given texts.
    It is used for benchmarking purposes.
    """

    def set_next_embedding(self, embedding: List[float]):
        self.next_embedding = embedding

    @override
    def get_embedding(self, text: str) -> List[float]:
        if self.next_embedding is None:
            raise ValueError("No next embedding set")
        return self.next_embedding
