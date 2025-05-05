from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines
    """

    @abstractmethod
    def create(self, prompt: str, system_prompt: str = None) -> str:
        """
        prompt: str - The prompt to create an answer for
        output_format: str - The optional output format to use for the response
        returns: str - The answer to the prompt
        """
        pass


class DummyInferenceEngine(InferenceEngine):
    """A simple fallback strategy that doesn't require any dependencies"""

    def __init__(self):
        super().__init__()

    def create(self, prompt: str, output_format: str = None) -> str:
        return f"[DUMMY RESPONSE] This is a placeholder response for: '{prompt}'"
