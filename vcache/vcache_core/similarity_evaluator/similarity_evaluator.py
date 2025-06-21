from abc import ABC, abstractmethod


class SimilarityEvaluator(ABC):
    """
    Abstract base class for evaluating similarity between answers.
    """

    def __init__(self):
        """
        Initialize similarity evaluator.
        """

    @abstractmethod
    def answers_similar(
        self,
        a: str,
        b: str,
        id_set_a: int = None,
        id_set_b: int = None,
    ) -> bool:
        """
        Determine if two answers are similar.

        Args:
            a: The first answer.
            b: The second answer.
            id_set_a: The id_set of the first answer.
            id_set_b: The id_set of the second answer.

        Returns:
            True if the answers are similar, False otherwise.
        """
        pass
