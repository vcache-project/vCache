from typing import Any, Optional

from openai import OpenAI as OpenAIClient
from typing_extensions import override

from vectorq.inference_engine.inference_engine import InferenceEngine


class OpenAIInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        temperature: float = 1,
        system_prompt: Optional[str] = None,
        **client_kwargs: Any,
    ):
        """
        Initialize the OpenAI inference engine.
        Args
            model_name: str - The name of the model to use by default.
            system_prompt: Optional[str] - The optional system prompt to use for the completion by default.
            temperature: float - The default temperature to use by default.
            **client_kwargs: Any - Additional keyword arguments to pass to the OpenAI client.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client_kwargs = client_kwargs
        self._client = None

    @property
    def client(self) -> OpenAIClient:
        """
        Lazily initialize the OpenAI client only when needed.
        Returns
            OpenAIClient - The OpenAI client.
        """
        if self._client is None:
            self._client = OpenAIClient(**self.client_kwargs)
        return self._client

    @override
    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a completion from the OpenAI client.
        Args
            prompt: str - The prompt to create a completion for.
            system_prompt: Optional[str] - The optional system prompt to use for the completion. It will override the system prompt in the OpenAIInferenceEngine if provided.
            **kwargs: Any - Additional keyword arguments to pass to the OpenAI chat completion API.
        Returns
            str - The completion from the OpenAI client.
        """
        try:
            # Build messages array with optional system prompt
            messages = []
            system_prompt = system_prompt or self.system_prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Prepare API call arguments
            args = {
                "model": kwargs.get("model", self.model_name),
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": messages,
                **kwargs,
            }
            completion = self.client.chat.completions.create(**args)
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error creating completion from OpenAI: {e}")
