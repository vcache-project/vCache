from vcache.vcache_core.cache.embedding_store.embedding_store import (
    EmbeddingMetadataObj,
)
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

    def answers_similar(
        self,
        a: str,
        b: str,
        metadata_a: EmbeddingMetadataObj = None,
        metadata_b: EmbeddingMetadataObj = None,
    ) -> bool:
        """
        Check if two answers are similar using LLM-judge comparison.

        Args:
            a: The first answer to compare.
            b: The second answer to compare.
            metadata_a: The metadata of the first answer (used for benchmark evaluation).
            metadata_b: The metadata of the second answer (used for benchmark evaluation).

        Returns:
            True if the answers are similar, False otherwise.
        """
        if not self.inference_engine:
            return False

        system_prompt: str = "You are a judge evaluating whether two answers are semantically equivalent. Respond with only 'YES' if they convey the same meaning, or 'NO' if they differ significantly."

        user_prompt: str = f"Answer 1: {a}\n\nAnswer 2: {b}\n\nAre these answers semantically equivalent?"

        try:
            response: str = (
                self.inference_engine.create(user_prompt, system_prompt).strip().upper()
            )
            return "YES" in response
        except Exception:
            return False
