from openai import OpenAI as OpenAIClient
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig

from vectorq.inference_engine.strategy import InferenceEngineStrategy


class OpenAI(InferenceEngineStrategy):

    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
        self.client = OpenAIClient()

    def create(self, prompt: str, output_format: str = None) -> str:
        try:
            messages = []
            if output_format:
                messages.append({"role": "system", "content": output_format})
            messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.vectorq_config._inference_engine_model_name,
                messages=messages,
                temperature=self.vectorq_config._inference_engine_temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error creating completion from OpenAI: {e}")
