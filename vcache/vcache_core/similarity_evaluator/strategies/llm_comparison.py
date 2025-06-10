from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self):
        super().__init__()

    def answers_similar(self, a: str, b: str) -> bool:
        if not self.inference_engine:
            return False

        system_prompt = "You are a judge evaluating whether two answers are semantically equivalent. Respond with only 'YES' if they convey the same meaning, or 'NO' if they differ significantly."

        user_prompt = f"Answer 1: {a}\n\nAnswer 2: {b}\n\nAre these answers semantically equivalent?"

        try:
            response = (
                self.inference_engine.create(user_prompt, system_prompt).strip().upper()
            )
            return response.startswith("YES")
        except Exception:
            return False
