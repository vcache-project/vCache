from enum import Enum
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.similarity_evaluator.strategy import SimilarityEvaluatorStrategy
from vectorq.vectorq_core.similarity_evaluator.strategies.string_comparison import StringComparison
from vectorq.vectorq_core.similarity_evaluator.strategies.embedding_comparison import EmbeddingComparison
from vectorq.vectorq_core.similarity_evaluator.strategies.llm_comparison import LLMComparison

class SimilarityEvaluatorType(Enum):
    STRING_COMPARISON = "string_comparison"
    EMBEDDING_COMPARISON = "embedding_comparison"
    LLM_COMPARISON = "llm_comparison"
    
class SimilarityEvaluator():

    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        self.strategy: SimilarityEvaluatorStrategy = None
        
        match self.vectorq_config._similarity_evaluator_type:
            case SimilarityEvaluatorType.STRING_COMPARISON:
                self.strategy: SimilarityEvaluatorStrategy = StringComparison()
            case SimilarityEvaluatorType.EMBEDDING_COMPARISON:
                self.strategy: SimilarityEvaluatorStrategy = EmbeddingComparison()
            case SimilarityEvaluatorType.LLM_COMPARISON:
                self.strategy: SimilarityEvaluatorStrategy = LLMComparison()
            case _:
                raise ValueError(f"Invalid similarity evaluator type")
            
    def answers_similar(self, a: str, b: str) -> bool:
        return self.strategy.answers_similar(a, b)
