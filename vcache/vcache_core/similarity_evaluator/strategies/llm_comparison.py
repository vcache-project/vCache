from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self):
        super().__init__()

    def answers_similar(self, a: str, b: str) -> bool:
        # TODO
        # @Alex: You can access the inference engine via:
        # self.inference_engine
        print("TODO: Embedding-based Answer similarity check not implemented")
        return False
