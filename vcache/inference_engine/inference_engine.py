from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    """

    @abstractmethod
    def create(self, prompt: str, system_prompt: str = None) -> str:
        """
        Create a response for the given prompt.

        Args:
            prompt: The prompt to create an answer for.
            system_prompt: The optional system prompt to use for the response.

        Returns:
            The answer to the prompt.
        """
        pass
