from typing import List
from vectorq.vectorq_core.bayesian_inference.strategy import LikelihoodFunctionStrategy

class Exponential(LikelihoodFunctionStrategy):
    
    def get_likelihood(self, values: List[float], x_sample: float) -> float:
        # TODO
        return 0.0
