import logging

from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    """
    LLM-based similarity evaluator for comparing answer similarity.
    """

    def __init__(self, inference_engine: InferenceEngine):
        """
        Initialize LLM comparison similarity evaluator.
        """
        super().__init__()
        self.inference_engine = inference_engine

    def answers_similar(
        self,
        a: str,
        b: str,
        id_set_a: int = None,
        id_set_b: int = None,
    ) -> bool:
        """
        Check if two answers are similar using LLM-judge comparison.

        Args:
            a: The first answer to compare.
            b: The second answer to compare.
            id_set_a: The id_set of the first answer.
            id_set_b: The id_set of the second answer.

        Returns:
            True if the answers are similar, False otherwise.
        """
        if not self.inference_engine:
            logging.warning("No inference engine provided. Returning False.")
            return False

        system_prompt: str = """
You are an expert judge evaluating whether two answers are semantically equivalent for caching purposes. 

Your task is to determine if two answers convey essentially the same meaning, even if they use different words, phrasing, or structure.

GUIDELINES:
- Focus on semantic meaning rather than exact wording
- Consider answers equivalent if they provide the same information
- Minor differences in phrasing, word choice, or formatting should be ignored
- Different examples that illustrate the same concept should be considered equivalent
- Answers with the same conclusion but different reasoning paths may be equivalent

EXAMPLE 1:
Answer 1: "The capital of France is Paris, which is located in the northern part of the country."
Answer 2: "Paris is the capital city of France."
Evaluation: YES

EXAMPLE 2:
Answer 1: "To solve this equation, multiply both sides by 2 to get x = 10."
Answer 2: "The solution is x = 5 after dividing both sides by 2."
Evaluation: NO
Respond with only "YES" if the answers are semantically equivalent, or "NO" if they differ significantly in meaning."""

        user_prompt: str = f"""
Answer 1: {a}
Answer 2: {b}

Are these answers semantically equivalent?"""

        try:
            response: str = (
                self.inference_engine.create(user_prompt, system_prompt).strip().upper()
            )
            return "YES" in response
        except Exception:
            return False
