from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    LLM-based similarity evaluator for comparing answer similarity.
    """

    def __init__(self):
        """
        Initialize LLM comparison similarity evaluator.
        """
        super().__init__()

    def answers_similar(self, a: str, b: str) -> bool:
        """
        Check if two answers are similar using LLM comparison.

        Args:
            a: The first answer to compare.
            b: The second answer to compare.

        Returns:
            True if the answers are similar, False otherwise.
        """
        # TODO
        # @Alex: You can access the inference engine via:
        # self.inference_engine
        print("TODO: Embedding-based Answer similarity check not implemented")
        return False
