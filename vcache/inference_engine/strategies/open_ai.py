from typing import Optional

from openai import OpenAI as OpenAIClient

from vcache.inference_engine.inference_engine import InferenceEngine


class OpenAIInferenceEngine(InferenceEngine):
    """
    OpenAI-based inference engine for generating responses using OpenAI's API.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        temperature: float = 1,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI inference engine.

        Args:
            model_name: Name of the OpenAI model to use.
            temperature: Temperature parameter for response generation.
            api_key: OpenAI API key for authentication.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self._client = None
        self.next_response = None

    def set_next_response(self, response: str):
        """
        Set the next response to be used for the next inference.
        """
        self.next_response = response

    @property
    def client(self) -> OpenAIClient:
        """
        Get the OpenAI client, initializing it lazily when needed.

        Returns:
            The OpenAI client instance.
        """
        if self._client is None:
            self._client = OpenAIClient(api_key=self.api_key, timeout=60.0)
        return self._client

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Create a response for the given prompt using OpenAI's API.

        Args:
            prompt: The prompt to create an answer for.
            system_prompt: The optional system prompt to use for the response.

        Returns:
            The answer to the prompt.
        """
        if self.next_response is not None:
            next_response = self.next_response
            self.next_response = None
            return next_response

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                timeout=60.0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            print(f"Error creating completion from OpenAI: {e}")
            if any(keyword in error_str for keyword in ["invalid_prompt", "invalid prompt", "flagged", "usage policy", "content policy", "safety", "harmful"]):
                return "I apologize, but I cannot provide a response to this prompt due to content policy restrictions."
            else:
                raise Exception(f"Error creating completion from OpenAI: {e}")
