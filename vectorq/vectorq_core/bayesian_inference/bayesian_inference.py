from enum import Enum
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.vectorq_core.bayesian_inference.strategy import LikelihoodFunctionStrategy
from vectorq.vectorq_core.bayesian_inference.strategies.exponential import Exponential
from vectorq.vectorq_core.bayesian_inference.strategies.sigmoid import Sigmoid
from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj

class LikelihoodFunctionType(Enum):
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    
class BayesianInference:
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
        self.strategy: LikelihoodFunctionStrategy = None
        
        match self.vectorq_config._likelihood_function_type:
            case LikelihoodFunctionType.EXPONENTIAL:
                self.strategy: LikelihoodFunctionStrategy = Exponential()
            case LikelihoodFunctionType.SIGMOID:
                self.strategy: LikelihoodFunctionStrategy = Sigmoid()
            case _:
                raise ValueError(f"Invalid likelihood function type")
    
    def update_posterior(self, is_correct: bool, metadata: EmbeddingMetadataObj) -> None:
        # TODO
        print("TODO: implement update_posterior")
        pass
    
    def get_posterior_likelihood(self, sample_likelihood: float, metadata: EmbeddingMetadataObj) -> float:
        # TODO
        print("TODO: implement get_posterior_likelihood")
        return 0.0
