from abc import ABC, abstractmethod
from typing import Any, Optional


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    """

    @abstractmethod
    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Infer a response from the inference engine.
        Args
            prompt: str - The prompt to create a response for.
            system_prompt: Optional[str] - The optional system prompt to use for the response.
            **kwargs: Any - Additional arguments to pass to the underlying inference engine (e.g., max_tokens, temperature, etc).
        Returns
            str - The response of the prompt.
        """
        pass
