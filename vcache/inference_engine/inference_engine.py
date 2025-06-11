from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    def create(self, prompt: str, system_prompt: str = None) -> str:
        """Create an answer for the given prompt.

        Args:
            prompt (str): The prompt to create an answer for.
            system_prompt (str, optional): The optional system prompt to use for the response.

        Returns:
            str: The answer to the prompt.
        """
        pass
