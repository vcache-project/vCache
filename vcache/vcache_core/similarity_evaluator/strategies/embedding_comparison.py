from vcache.vcache_core.cache.embedding_store.embedding_store import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class EmbeddingComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    Embedding-based similarity evaluator for comparing answer similarity.
    """

    def __init__(self):
        """
        Initialize embedding comparison similarity evaluator.
        """
        super().__init__()

    def answers_similar(
        self,
        a: str,
        b: str,
        metadata_a: EmbeddingMetadataObj = None,
        metadata_b: EmbeddingMetadataObj = None,
    ) -> bool:
        """
        Check if two answers are similar using embedding comparison.

        Args:
            a: The first answer to compare.
            b: The second answer to compare.
            metadata_a: The metadata of the first answer (used for benchmark evaluation).
            metadata_b: The metadata of the second answer (used for benchmark evaluation).

        Returns:
            True if the answers are similar, False otherwise.
        """
        # TODO
        print("TODO: Embedding-based Answer similarity check not implemented")
        return False
