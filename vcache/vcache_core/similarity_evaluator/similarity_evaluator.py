from abc import ABC, abstractmethod

from vcache.inference_engine import InferenceEngine


class SimilarityEvaluator(ABC):
    def __init__(self):
        self.inference_engine: InferenceEngine = None

    @abstractmethod
    def answers_similar(self, a: str, b: str) -> bool:
        """
        Evaluates the similarity between two answers.

        Args:
            a: str - The first answer
            b: str - The second answer

        Returns:
            bool - True if the answers are similar, False otherwise
        """
        pass

    def set_inference_engine(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
