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
        id_set_a: int = None,
        id_set_b: int = None,
    ) -> bool:
        """
        Check if two answers are similar using embedding comparison.

        Args:
            a: The first answer to compare.
            b: The second answer to compare.
            id_set_a: The id_set of the first answer.
            id_set_b: The id_set of the second answer.

        Returns:
            True if the answers are similar, False otherwise.
        """
        # TODO
        print("TODO: Embedding-based Answer similarity check not implemented")
        return False
