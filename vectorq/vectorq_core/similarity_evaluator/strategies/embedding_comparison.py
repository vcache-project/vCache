from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator

class EmbeddingComparisonSimilarityEvaluator(SimilarityEvaluator):
    
    def __init__(self):
        super().__init__()
    
    def answers_similar(self, a: str, b: str) -> bool:
        # TODO
        print("TODO: Embedding-based Answer similarity check not implemented")
        return False
