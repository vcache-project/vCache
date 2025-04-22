from openai import OpenAI as OpenAIClient

from vectorq.inference_engine.inference_engine import InferenceEngine


class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model_name: str, temperature: float = 1):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAIClient()

    def create(self, prompt: str, output_format: str = None) -> str:
        try:
            messages = []
            if output_format:
                messages.append({"role": "system", "content": output_format})
            messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error creating completion from OpenAI: {e}")
