from vcache.vcache_core.cache.embedding_store.embedding_store import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class StringComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    String-based similarity evaluator that compares normalized text.
    """

    def __init__(self):
        """
        Initialize string comparison similarity evaluator.
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
        Determine if two answers are similar using string comparison.

        Args:
            a: The first answer.
            b: The second answer.
            metadata_a: The metadata of the first answer (used for benchmark evaluation).
            metadata_b: The metadata of the second answer (used for benchmark evaluation).

        Returns:
            True if the normalized strings are equal, False otherwise.
        """
        answer_a = (
            str(a)
            .strip()
            .rstrip(".")
            .lower()
            .replace('"', "")
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
        )
        answer_b = (
            str(b)
            .strip()
            .rstrip(".")
            .lower()
            .replace('"', "")
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
        )
        return answer_a == answer_b
