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

    @property
    def client(self) -> OpenAIClient:
        """
        Get the OpenAI client, initializing it lazily when needed.

        Returns:
            The OpenAI client instance.
        """
        if self._client is None:
            self._client = OpenAIClient(api_key=self.api_key)
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
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error creating completion from OpenAI: {e}")
