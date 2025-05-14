from typing import Optional

from openai import OpenAI as OpenAIClient

from vectorq.inference_engine.inference_engine import InferenceEngine


class OpenAIInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano-2025-04-14",
        temperature: float = 0,
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> OpenAIClient:
        """Lazily initialize the OpenAI client only when needed."""
        if self._client is None:
            self._client = OpenAIClient(api_key=self.api_key)
        return self._client

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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
