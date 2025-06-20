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
        id_set_a: int = None,
        id_set_b: int = None,
    ) -> bool:
        """
        Determine if two answers are similar using string comparison.

        Args:
            a: The first answer.
            b: The second answer.
            id_set_a: The id_set of the first answer.
            id_set_b: The id_set of the second answer.

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
