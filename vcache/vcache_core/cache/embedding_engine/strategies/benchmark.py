from typing import List

from typing_extensions import override

from vcache.vcache_core.cache.embedding_engine.embedding_engine import (
    EmbeddingEngine,
)


class BenchmarkEmbeddingEngine(EmbeddingEngine):
    """
    An embedding engine implementation that returns pre-computed embeddings for given texts.
    It is used for benchmarking purposes.
    """

    def set_next_embedding(self, embedding: List[float]):
        """
        Set the next embedding to be returned by get_embedding.

        Args:
            embedding: The embedding vector to return on next call.
        """
        self.next_embedding = embedding

    @override
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the pre-set embedding vector.

        Args:
            text: The text to embed (ignored in benchmark mode).

        Returns:
            The pre-set embedding vector.

        Raises:
            ValueError: If no embedding has been set.
        """
        if self.next_embedding is None:
            raise ValueError("No next embedding set")
        return self.next_embedding
