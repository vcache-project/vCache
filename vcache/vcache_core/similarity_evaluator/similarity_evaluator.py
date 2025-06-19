from abc import ABC, abstractmethod

from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.embedding_store.embedding_store import (
    EmbeddingMetadataObj,
)


class SimilarityEvaluator(ABC):
    """
    Abstract base class for evaluating similarity between answers.
    """

    def __init__(self):
        """
        Initialize similarity evaluator.
        """
        self.inference_engine: InferenceEngine = None

    @abstractmethod
    def answers_similar(
        self,
        a: str,
        b: str,
        metadata_a: EmbeddingMetadataObj = None,
        metadata_b: EmbeddingMetadataObj = None,
    ) -> bool:
        """
        Determine if two answers are similar.

        Args:
            a: The first answer.
            b: The second answer.
            metadata_a: The metadata of the first answer (used for benchmark evaluation).
            metadata_b: The metadata of the second answer (used for benchmark evaluation).

        Returns:
            True if the answers are similar, False otherwise.
        """
        pass

    def set_inference_engine(self, inference_engine: InferenceEngine):
        """
        Set the inference engine for the similarity evaluator.

        Args:
            inference_engine: The inference engine to use.
        """
        self.inference_engine = inference_engine
