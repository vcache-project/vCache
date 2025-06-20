import logging

from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class BenchmarkComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    Benchmark-based similarity evaluator that compares answers based on their id_set.

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
        id_set_a: int,
        id_set_b: int,
    ) -> bool:
        """
        Check if two answers are similar by comparing their id_set metadata.

        This method determines similarity based on whether both answers belong to the
        same id_set, which is used in benchmark scenarios to identify if responses
        are from the same correctness group (both correct or both incorrect).

        Args:
            a: The first answer to compare (not used in this implementation).
            b: The second answer to compare (not used in this implementation).
            id_set_a: The id_set of the first answer.
            id_set_b: The id_set of the second answer.

        Returns:
            True if both answers belong to the same id_set, False otherwise.
        """
        if id_set_a == -1 or id_set_b == -1:
            logging.warning(
                f"BenchmarkComparisonSimilarityEvaluator - Verify your Configuration - id_set_a: {id_set_a}, id_set_b: {id_set_b}"
            )
            return False

        return id_set_a == id_set_b
