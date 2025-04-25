from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class StringComparisonSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self):
        super().__init__()

    def answers_similar(self, a: str, b: str) -> bool:
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
