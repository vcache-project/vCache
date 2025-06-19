from vcache.vcache_core.cache.embedding_store.embedding_store import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class BenchmarkComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    Benchmark-based similarity evaluator that compares answers based on their id_set metadata.

    This evaluator is specifically designed for benchmark evaluation scenarios where
    answers are considered similar if they belong to the same id_set. This allows
    for ground truth comparison during benchmarking by grouping correct and incorrect
    responses together.
    """

    def __init__(self):
        """
        Initialize benchmark comparison similarity evaluator.
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
        Check if two answers are similar by comparing their id_set metadata.

        This method determines similarity based on whether both answers belong to the
        same id_set, which is used in benchmark scenarios to identify if responses
        are from the same correctness group (both correct or both incorrect).

        Args:
            a: The first answer to compare (not used in this implementation).
            b: The second answer to compare (not used in this implementation).
            metadata_a: The metadata of the first answer containing the id_set.
            metadata_b: The metadata of the second answer containing the id_set.

        Returns:
            True if both answers belong to the same id_set, False otherwise.
        """
        return metadata_a.id_set == metadata_b.id_set
